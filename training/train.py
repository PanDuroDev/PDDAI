import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import math
import time
import tempfile
import shutil
from collections import deque
from torch.amp import autocast, GradScaler
from utils.auto_config import dynamic_adjust_threads
from utils.muon import Muon


def save_checkpoint(state, checkpoint_path):
    tmp = tempfile.NamedTemporaryFile(
        dir=os.path.dirname(checkpoint_path),
        suffix='.tmp',
        delete=False
    )
    try:
        torch.save(state, tmp.name)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp.close()
        shutil.move(tmp.name, checkpoint_path)
    except Exception:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)
        raise


def load_checkpoint(checkpoint_path, device, model, optimizer, scheduler, scaler=None, grad_accum_steps=1):
    if not os.path.exists(checkpoint_path):
        return 0, 0, 0

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        print(f"Cannot read checkpoint: {e}")
        return 0, 0, 0

    try:
        result = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if result.missing_keys:
            print(f"  WARNING: Missing keys: {result.missing_keys}")
        if result.unexpected_keys:
            print(f"  WARNING: Unexpected keys: {result.unexpected_keys}")
        if model.tied_embeddings:
            model.head.weight = model.token_emb.weight
    except Exception as e:
        print(f"Model weights mismatch: {e}")
        return 0, 0, 0

    try:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    except Exception as e:
        print(f"Optimizer state incompatible: {e}")
        print("Continuing with fresh optimizer (weights loaded).")

    if 'scheduler_state_dict' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except Exception as e:
            print(f"Scheduler state incompatible: {e}")
            print("Continuing with fresh scheduler.")

    if scaler and 'scaler_state_dict' in checkpoint:
        try:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        except Exception as e:
            print(f"Scaler state incompatible: {e}")

    if 'rng_state' in checkpoint:
        try:
            torch.set_rng_state(checkpoint['rng_state'].cpu().byte())
            if torch.cuda.is_available():
                cuda_rng = [s.cpu().byte() for s in checkpoint['cuda_rng_state']]
                torch.cuda.set_rng_state_all(cuda_rng)
        except Exception as e:
            print(f"RNG restore failed: {e}")

    start_epoch = checkpoint.get('epoch', 0)
    start_batch = checkpoint.get('batch', 0)
    global_step = checkpoint.get('global_step', 0)
    ckpt_version = checkpoint.get('_ckpt_version', 1)
    if ckpt_version < 2:
        global_step = global_step // max(1, grad_accum_steps)
    print(f"Resumed: epoch {start_epoch}, batch {start_batch}, opt_step {global_step}")
    return start_epoch, start_batch, global_step


def make_optimizer(model, use_muon, muon_lr, adamw_lr):
    """Hybrid Muon+AdamW optimizer. Muon for 2D weights, AdamW for 1D/biases/embeds."""
    if not use_muon:
        return optim.AdamW(model.parameters(), lr=adamw_lr, betas=(0.9, 0.95), weight_decay=0.01)

    hidden_2d = []
    hidden_1d = []
    embed_params = []
    head_params = []

    seen = set()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.data_ptr() in seen:
            continue
        seen.add(p.data_ptr())
        if 'token_emb' in name:
            embed_params.append(p)
        elif 'head' in name:
            head_params.append(p)
        elif p.ndim >= 2:
            hidden_2d.append(p)
        else:
            hidden_1d.append(p)

    param_groups = [
        {'params': hidden_2d, 'use_muon': True, 'lr': muon_lr, 'weight_decay': 0.01},
        {'params': hidden_1d, 'use_muon': False, 'lr': adamw_lr, 'betas': (0.9, 0.95), 'weight_decay': 0.01},
        {'params': embed_params, 'use_muon': False, 'lr': adamw_lr, 'betas': (0.9, 0.95), 'weight_decay': 0.01},
        {'params': head_params, 'use_muon': False, 'lr': adamw_lr, 'betas': (0.9, 0.95), 'weight_decay': 0.01},
    ]
    return Muon(param_groups)


def _format_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def _progress_bar(current, total, loss, lr, speed, eta, elapsed, tok_per_sec, bar_len=30):
    frac = current / total
    filled = int(bar_len * frac)
    bar = '#' * filled + '.' * (bar_len - filled)
    pct = frac * 100
    ms_batch = 1000 / max(0.001, speed)
    return (f"  |{bar}| {pct:6.2f}% | batch {current:>5}/{total} "
            f"| loss {loss:.4f} | lr {lr:.1e} "
            f"| {speed:.2f} b/s | {tok_per_sec:.0f} t/s | {ms_batch:.0f}ms | ETA {eta}")


def _build_model_config(model):
    first_block = model.blocks[0]
    return {
        'embed_dim': model.embed_dim,
        'num_heads': first_block.attn.num_heads,
        'num_kv_heads': first_block.attn.num_kv_heads,
        'ff_hidden_dim': first_block.ffn.gate.weight.shape[0],
        'num_layers': len(model.blocks),
        'max_len': model.rope_cos.shape[0],
        'rope_theta': 2500.0,
        'vocab_size': model.head.out_features,
        'tied_embeddings': model.tied_embeddings,
        'refresh_layers': [int(k.split('.')[1]) for k in model.state_dict() if 'refresh.gate.weight' in k],
    }


def _build_checkpoint(model, optimizer, scheduler, scaler, epoch, batch, opt_step, vocab_size):
    ckpt = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'batch': batch,
        'global_step': opt_step,
        '_ckpt_version': 2,
        'vocab_size': vocab_size,
        'tied_embeddings': model.tied_embeddings,
        'model_config': _build_model_config(model),
        'rng_state': torch.get_rng_state().cpu(),
    }
    if scaler:
        ckpt['scaler_state_dict'] = scaler.state_dict()
    if torch.cuda.is_available():
        ckpt['cuda_rng_state'] = torch.cuda.get_rng_state_all()
    return ckpt


def _build_inference_state(model, epoch, opt_step, val_loss):
    return {
        'model_state_dict': model.state_dict(),
        'vocab_size': model.head.out_features,
        'tied_embeddings': model.tied_embeddings,
        'model_config': _build_model_config(model),
        'epoch': epoch,
        'opt_step': opt_step,
        'val_loss': val_loss,
        'timestamp': time.time(),
        'format_version': 1,
    }


def save_model_for_inference(model, path, epoch, opt_step, val_loss):
    state = _build_inference_state(model, epoch, opt_step, val_loss)
    tmp = tempfile.NamedTemporaryFile(
        dir=os.path.dirname(path),
        suffix='.tmp',
        delete=False
    )
    try:
        torch.save(state, tmp.name)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp.close()
        shutil.move(tmp.name, path)
    except Exception:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)
        raise


@torch.inference_mode()
def validate(model, val_dataloader, device, criterion, seq_len=512):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for batch_data, batch_targets in val_dataloader:
        batch_data = batch_data.to(device, non_blocking=True)
        batch_targets = batch_targets.to(device, non_blocking=True)
        output = model(batch_data)
        V = output.shape[-1]
        loss = criterion(output.view(-1, V), batch_targets.view(-1))
        total_loss += loss.item()
        total_batches += 1
    model.train()
    return total_loss / max(1, total_batches)


def train_model(model, dataloader, device, val_dataloader=None, seq_len=512,
                epochs=10, lr=1e-4, grad_accum_steps=1,
                use_amp=True, use_checkpoint=False, save_every=1,
                label_smoothing=0.0, use_unlikelihood=False, use_muon=False,
                warmup_steps=None, muon_lr=None, adamw_lr=None,
                gradient_clipping=1.0,
                save_last_model=True, save_best_model=True,
                save_every_n_batches=0):
    save_dir = "saved_model"
    os.makedirs(save_dir, exist_ok=True)

    model.to(device)
    seen_ptrs = set()
    unique_params = []
    for p in model.parameters():
        if p.data_ptr() not in seen_ptrs:
            seen_ptrs.add(p.data_ptr())
            unique_params.append(p)

    optimizer = make_optimizer(model, use_muon=use_muon, muon_lr=muon_lr, adamw_lr=adamw_lr)
    total_batches = len(dataloader)
    total_opt_steps = (epochs * total_batches + grad_accum_steps - 1) // grad_accum_steps

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    amp_device = 'cuda' if 'cuda' in str(device) else 'cpu'
    use_amp_enabled = use_amp and amp_device == 'cuda'
    scaler = GradScaler(amp_device) if use_amp_enabled else None

    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: min(s / warmup_steps, 1.0) if s < warmup_steps
        else 0.5 * (1 + math.cos(math.pi * (s - warmup_steps) / max(1, total_opt_steps - warmup_steps)))
    )

    checkpoint_path = os.path.join(save_dir, "checkpoint.pt")
    last_model_path = os.path.join(save_dir, "last_model.pt")
    best_model_path = os.path.join(save_dir, "best_model.pt")
    start_epoch, start_batch, opt_step = load_checkpoint(
        checkpoint_path, device, model, optimizer, scheduler, scaler,
        grad_accum_steps=grad_accum_steps
    )

    if start_epoch >= epochs:
        print(f"Model already trained for {start_epoch} epochs (target: {epochs}).")
        return

    best_val_loss = float('inf')
    for candidate in [best_model_path, os.path.join(save_dir, "checkpoint_best.pt")]:
        if os.path.exists(candidate):
            try:
                best_ckpt = torch.load(candidate, map_location='cpu')
                best_val_loss = best_ckpt.get('val_loss', float('inf'))
                break
            except Exception:
                pass

    model.train()
    optimizer.zero_grad()
    remaining = epochs - start_epoch
    print(f"Training: {remaining} more epochs, {total_batches} batches/epoch, grad_accum={grad_accum_steps}")

    N = dataloader.dataset[0][0].size(0)
    ul_tril = torch.tril(torch.ones(N, N, dtype=torch.bool, device=device), diagonal=-1) if use_unlikelihood else None

    t_start = time.time()
    epoch = None
    batch_idx = None
    batch_count = start_epoch * total_batches + start_batch
    batch_timestamps = deque(maxlen=100)
    last_log_time = 0.0

    try:
        for epoch in range(start_epoch, epochs):
            epoch_loss_gpu = torch.zeros(1, device=device)
            data_iter = iter(dataloader)

            for batch_idx in range(start_batch if epoch == start_epoch else 0, total_batches):
                batch_count += 1
                try:
                    batch_data, batch_targets = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch_data, batch_targets = next(data_iter)

                batch_data = batch_data.to(device, non_blocking=True)
                batch_targets = batch_targets.to(device, non_blocking=True)

                with autocast(device_type=amp_device, enabled=use_amp_enabled):
                    output = model(batch_data, use_checkpoint=use_checkpoint)
                    _, _, V = output.shape
                    loss = criterion(output.view(-1, V), batch_targets.view(-1))

                if use_unlikelihood:
                    probs = F.softmax(output.float().view(-1, V), dim=-1)
                    eq = batch_targets.unsqueeze(2) == batch_targets.unsqueeze(1)
                    repeated = (eq & ul_tril.unsqueeze(0)).any(dim=2)
                    ul_mask = repeated.view(-1)
                    if ul_mask.any():
                        ul_loss = probs[ul_mask, batch_targets.view(-1)[ul_mask]].mean()
                        loss = loss + 0.1 * ul_loss

                loss = loss / grad_accum_steps

                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (batch_idx + 1) % grad_accum_steps == 0:
                    if scaler:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(unique_params, gradient_clipping)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(unique_params, gradient_clipping)
                        optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    opt_step += 1

                    if save_every_n_batches > 0 and opt_step % save_every_n_batches == 0:
                        save_model_for_inference(
                            model, last_model_path,
                            epoch + 1, opt_step, None
                        )

                epoch_loss_gpu += loss.detach() * grad_accum_steps

                now = time.time()
                batch_timestamps.append(now)
                if now - last_log_time >= 1.0 or (batch_idx + 1) == total_batches:
                    last_log_time = now
                    if len(batch_timestamps) >= 2:
                        window_speed = len(batch_timestamps) / max(0.001, batch_timestamps[-1] - batch_timestamps[0])
                    else:
                        window_speed = batch_count / max(1, now - t_start)
                    remaining_batches = (epochs - epoch - 1) * total_batches + (total_batches - batch_idx - 1)
                    eta_remaining = remaining_batches / max(0.001, window_speed)
                    current_lr = optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else scheduler.get_last_lr()[0]
                    tok_per_sec_now = window_speed * dataloader.batch_size * seq_len
                    print(_progress_bar(
                        batch_idx + 1, total_batches,
                        loss.item() * grad_accum_steps,
                        current_lr,
                        window_speed,
                        _format_time(eta_remaining),
                        now - t_start,
                        tok_per_sec_now,
                    ), end='\r')
                    dynamic_adjust_threads()

            avg_loss = epoch_loss_gpu.item() / total_batches
            elapsed = time.time() - t_start
            if len(batch_timestamps) >= 2:
                epoch_speed = len(batch_timestamps) / max(0.001, batch_timestamps[-1] - batch_timestamps[0])
            else:
                epoch_speed = batch_count / max(1, elapsed)
            tok_per_sec = epoch_speed * dataloader.batch_size * seq_len

            val_loss = None
            if val_dataloader is not None:
                val_loss = validate(model, val_dataloader, device, criterion, seq_len)
            current_lr = optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else scheduler.get_last_lr()[0]
            is_best = val_loss is not None and val_loss < best_val_loss

            parts = [f"Epoch {epoch+1}/{epochs}"]
            parts.append(f"Loss: {avg_loss:.4f}")
            if val_loss is not None:
                parts.append(f"Val: {val_loss:.4f}")
                parts.append(f"(best {best_val_loss:.4f})" if not is_best else f"(best {val_loss:.4f})")
            parts.append(f"step {opt_step}")
            parts.append(f"lr {current_lr:.1e}")
            parts.append(f"{tok_per_sec:.0f} t/s")
            if 'cuda' in str(device):
                vram_alloc = torch.cuda.memory_allocated(device) / 1e9
                vram_reserv = torch.cuda.memory_reserved(device) / 1e9
                parts.append(f"VRAM {vram_alloc:.1f}/{vram_reserv:.1f}GB")
            parts.append(f"elapsed {_format_time(elapsed)}")
            print(" | ".join(parts))

            if is_best:
                best_val_loss = val_loss

            if save_last_model:
                save_model_for_inference(
                    model, last_model_path,
                    epoch + 1, opt_step, val_loss
                )
                print(f"  Inference model saved: {last_model_path}")

            if save_best_model and val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                save_model_for_inference(
                    model, best_model_path,
                    epoch + 1, opt_step, val_loss
                )
                print(f"  Best model saved (val_loss={val_loss:.4f})")

            if (epoch + 1) % save_every == 0:
                ckpt = _build_checkpoint(
                    model, optimizer, scheduler, scaler,
                    epoch + 1, 0, opt_step, model.head.out_features
                )
                ckpt['best_val_loss'] = best_val_loss
                save_checkpoint(ckpt, checkpoint_path)
                print(f"  Checkpoint saved: {checkpoint_path}")

    except KeyboardInterrupt:
        if epoch is not None and batch_idx is not None:
            print("\nSaving interrupted checkpoint...")
            ckpt = _build_checkpoint(
                model, optimizer, scheduler, scaler,
                epoch, batch_idx, opt_step, model.head.out_features
            )
            ckpt['best_val_loss'] = best_val_loss
            save_checkpoint(ckpt, checkpoint_path)
            if save_last_model:
                save_model_for_inference(
                    model, last_model_path,
                    epoch + 1, opt_step, None
                )
            print(f"Interrupted at epoch {epoch}, batch {batch_idx}")
            print(f"Checkpoint saved: {checkpoint_path}")
        else:
            print("\nInterrupted before training started.")
        raise
