#perception.py

"""
Policy 11 perception: an AlexNet convolutional stack that looks at the overhead camera image and
outputs the beaker and dish XY positions in the robot frame.

The image is taken once per episode, before the arm moves in, with the arm in a parked (leaned back)
pose so that it hides neither container. liquid_transfer_m3_v11.py feeds the four predicted numbers
to the policy in place of the ground-truth container positions that Policy 10 was given.

Training is supervised on images rendered from the same MuJoCo scene (fresh images every batch, so
there is no fixed dataset and nothing to overfit to).

Run from this folder:
    PY=../model_3_mujoco_ppo_known_positions/.venv/bin/python
    Train:    $PY perception.py --mode train --steps 6000
    Evaluate: $PY perception.py --mode eval --checkpoint perception.pt
    Preview:  $PY perception.py --mode preview        (writes camera_preview_*.png)
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE_XML = os.path.join(HERE, "liquid_transfer_scene.xml")

CAMERA = "overhead"
IMG_W, IMG_H = 256, 192               # 4:3 so the whole placement range is in view (about 4.6 mm per pixel)
BEAKER_NOMINAL = np.array([0.4, -0.2, 0.06])
DISH_NOMINAL = np.array([0.4, 0.2, 0.01])
XY_SCALE = 0.2                        # network output = (xy - nominal xy) / XY_SCALE
MIN_REACH_M = 0.29
MIN_SEPARATION_M = 0.18

HOME_QPOS = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04])
# Parked pose for the snapshot: upper arm leaned back, forearm folded. Every link is behind x = 0.25 m and
# no container placement in the task range is hidden from the overhead camera.
PARK_QPOS = np.array([0.0, -1.2, 0.0, -2.6, 0.0, 1.571, 0.785, 0.04, 0.04])
LOOK_QPOS = {"park": PARK_QPOS, "home": HOME_QPOS}


class ContainerLocator(nn.Module):
    """AlexNet convolutional layers (Krizhevsky et al. 2012, torchvision layout) + a small regression head.

    Input:  uint8 or float image batch (N, H, W, 3) or (N, 3, H, W), values 0-255.
    Output: (N, 4) = beaker x, beaker y, dish x, dish y in metres, robot frame.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # 256 x 192 input -> 256 channels x 5 x 7. AlexNet's two 4096-unit layers are replaced by one
        # 256-unit layer: the output is four coordinates, not 1000 classes.
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(256 * 5 * 7, 256), nn.ReLU(inplace=True), nn.Linear(256, 4))
        self.register_buffer("nominal_xy", torch.tensor([*BEAKER_NOMINAL[:2], *DISH_NOMINAL[:2]], dtype=torch.float32))

    def forward_normalized(self, img):
        if img.shape[-1] == 3:
            img = img.permute(0, 3, 1, 2)
        x = img.float() / 255.0 - 0.5
        return self.head(self.features(x))

    def forward(self, img):
        return self.forward_normalized(img) * XY_SCALE + self.nominal_xy


def load_locator(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    net = ContainerLocator().to(device)
    net.load_state_dict(ckpt.get("perception_state_dict", ckpt))
    net.eval()
    return net


class OverheadCamera:
    """Renders the overhead camera for a MuJoCo model. Create and use it on one thread."""

    def __init__(self, model):
        import mujoco
        self._mj = mujoco
        self.model = model
        self.renderer = mujoco.Renderer(model, height=IMG_H, width=IMG_W)

    def render(self, data):
        self.renderer.update_scene(data, camera=CAMERA)
        return self.renderer.render().copy()

    def snapshot(self, data, look_qpos):
        """Image with the arm at look_qpos and the containers where they are; leaves data.qpos unchanged."""
        mj = self._mj
        saved = data.qpos.copy()
        data.qpos[:9] = look_qpos
        mj.mj_forward(self.model, data)
        img = self.render(data)
        data.qpos[:] = saved
        mj.mj_forward(self.model, data)
        return img

    def close(self):
        try:
            self.renderer.close()
        except Exception:
            pass


def sample_layouts(rng, n, half_range):
    """Container XY positions with the task's validity rules (min reach, min separation)."""
    b = BEAKER_NOMINAL[:2] + rng.uniform(-half_range, half_range, size=(n, 2))
    d = DISH_NOMINAL[:2] + rng.uniform(-half_range, half_range, size=(n, 2))
    for _ in range(100):
        bad = (np.linalg.norm(b, axis=1) < MIN_REACH_M) | (np.linalg.norm(d, axis=1) < MIN_REACH_M)
        bad |= np.linalg.norm(b - d, axis=1) < MIN_SEPARATION_M
        if not bad.any():
            break
        k = int(bad.sum())
        b[bad] = BEAKER_NOMINAL[:2] + rng.uniform(-half_range, half_range, size=(k, 2))
        d[bad] = DISH_NOMINAL[:2] + rng.uniform(-half_range, half_range, size=(k, 2))
    return b, d


class SnapshotSampler:
    """Random labelled snapshots: (images uint8 (n, H, W, 3), labels (n, 4) in metres)."""

    def __init__(self, seed, half_range, look="mix"):
        import mujoco
        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data = mujoco.MjData(self.model)
        self.cam = OverheadCamera(self.model)
        self.rng = np.random.default_rng(seed)
        self.half_range, self.look = half_range, look
        self.beaker_mocap = self.model.body_mocapid[self.model.body("beaker").id]
        self.dish_mocap = self.model.body_mocapid[self.model.body("petri_dish").id]
        self.jr = self.model.jnt_range[:9].copy()

    def batch(self, n):
        mj, rng = self._mj, self.rng
        b, d = sample_layouts(rng, n, self.half_range)
        imgs = np.zeros((n, IMG_H, IMG_W, 3), dtype=np.uint8)
        for i in range(n):
            look = self.look if self.look != "mix" else ("park" if rng.random() < 0.6 else "home")
            # Joint noise: the home reset uses +-0.1 rad; the parked arm is assumed to be within +-0.05 rad.
            noise = rng.uniform(-1.0, 1.0, size=9) * (0.05 if look == "park" else 0.1)
            self.data.qpos[:9] = np.clip(LOOK_QPOS[look] + noise, self.jr[:, 0], self.jr[:, 1])
            self.data.mocap_pos[self.beaker_mocap] = [b[i, 0], b[i, 1], BEAKER_NOMINAL[2]]
            self.data.mocap_pos[self.dish_mocap] = [d[i, 0], d[i, 1], DISH_NOMINAL[2]]
            mj.mj_forward(self.model, self.data)
            imgs[i] = self.cam.render(self.data)
        return imgs, np.concatenate([b, d], axis=1).astype(np.float32)


def augment(imgs, rng):
    """Brightness, colour balance and sensor noise, so the network does not depend on exact pixel values."""
    x = torch.from_numpy(imgs).float()
    n = x.shape[0]
    gain = torch.from_numpy(rng.uniform(0.8, 1.2, size=(n, 1, 1, 1)) * rng.uniform(0.95, 1.05, size=(n, 1, 1, 3))).float()
    offset = torch.from_numpy(rng.uniform(-12.0, 12.0, size=(n, 1, 1, 1))).float()
    x = x * gain + offset + torch.randn_like(x) * 3.0
    return x.clamp_(0.0, 255.0)


def error_report(net, sampler, n, tag):
    errs = []
    with torch.no_grad():
        for _ in range(n // 250):
            imgs, labels = sampler.batch(250)
            pred = net(torch.from_numpy(imgs)).numpy()
            e = (pred - labels).reshape(-1, 2, 2)
            errs.append(np.linalg.norm(e, axis=2))
    errs = np.concatenate(errs) * 1000.0  # mm, columns: beaker, dish
    for j, name in enumerate(("beaker", "dish")):
        e = errs[:, j]
        print(f"[{tag}] {name}: mean {e.mean():.2f} mm, median {np.median(e):.2f}, p95 {np.percentile(e, 95):.2f}, "
              f"p99 {np.percentile(e, 99):.2f}, max {e.max():.2f}, >10 mm: {(e > 10).mean():.2%}  (n={len(e)})")
    return errs


def main():
    p = argparse.ArgumentParser(description="Policy 11 perception (AlexNet container locator).")
    p.add_argument("--mode", choices=["train", "eval", "preview"], default="eval")
    p.add_argument("--checkpoint", type=str, default=os.path.join(HERE, "perception.pt"))
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--train_range", type=float, default=0.20,
                   help="Half-range of container placement in training images (task range is 0.175).")
    p.add_argument("--eval_range", type=float, default=0.175)
    p.add_argument("--eval_n", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=0)
    args = p.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    if args.mode == "preview":
        from PIL import Image
        s = SnapshotSampler(args.seed, args.eval_range, look="park")
        for look in ("park", "home"):
            s.look = look
            s.rng = np.random.default_rng(args.seed)
            imgs, labels = s.batch(4)
            Image.fromarray(np.concatenate(list(imgs), axis=1)).save(os.path.join(HERE, f"camera_preview_{look}.png"))
            print(f"[Preview] camera_preview_{look}.png  labels (beaker xy, dish xy): {labels.round(3).tolist()}")
        return

    if args.mode == "train":
        net = ContainerLocator()
        sampler = SnapshotSampler(args.seed, args.train_range, look="mix")
        aug_rng = np.random.default_rng(args.seed + 1)
        opt = torch.optim.Adam(net.parameters(), lr=args.lr)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
        nominal = net.nominal_xy.numpy()
        t0 = time.perf_counter(); running = 0.0
        net.train()
        for step in range(1, args.steps + 1):
            imgs, labels = sampler.batch(args.batch)
            target = torch.from_numpy((labels - nominal) / XY_SCALE)
            loss = nn.functional.smooth_l1_loss(net.forward_normalized(augment(imgs, aug_rng)), target, beta=0.02)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            running += loss.item()
            if step % 100 == 0:
                print(f"[Train] step {step}/{args.steps} loss {running / 100:.5f} lr {sched.get_last_lr()[0]:.2e} "
                      f"elapsed {time.perf_counter() - t0:.0f}s", flush=True)
                running = 0.0
            if step % 1000 == 0 or step == args.steps:
                torch.save({"perception_state_dict": net.state_dict(), "img_hw": (IMG_H, IMG_W), "steps": step},
                           args.checkpoint)
        print(f"[Train] saved {args.checkpoint}")
        net.eval()
    else:
        net = load_locator(args.checkpoint)

    # Held-out check: different seed, the task's placement range, no augmentation.
    for look in ("park", "home"):
        error_report(net, SnapshotSampler(args.seed + 1000, args.eval_range, look=look), args.eval_n, f"Eval look={look}")


if __name__ == "__main__":
    main()
