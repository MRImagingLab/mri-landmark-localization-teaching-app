# app_centroid_net.py
# Streamlit app for centroid prediction using DSNT-trained TinyUNetLogits
#
# UI rewrite notes:
#   - Keeps the existing working model and utility functions unchanged
#   - Adds a built-in demo preview when no upload is provided
#   - Shows a clean top preview for print/screenshot purposes
#   - Uses compact figure panels first instead of a large mostly-black canonical panel
#   - Moves teaching expanders below the main visual outputs

from io import BytesIO
from pathlib import Path
from collections import OrderedDict

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import streamlit as st
from PIL import Image, ImageDraw

# ============================================================
# Model definitions (UNCHANGED)
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class FeatureHook:
    """Capture intermediate activations by module name."""
    def __init__(self, model: nn.Module, layer_names):
        self.model = model
        self.layer_names = set(layer_names)
        self.handles = []
        self.feats = OrderedDict()

        for name, module in self.model.named_modules():
            if name in self.layer_names:
                h = module.register_forward_hook(self._make_hook(name))
                self.handles.append(h)

    def _make_hook(self, name):
        def hook(module, inp, out):
            self.feats[name] = out.detach()
        return hook

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []


class TinyUNetLogits(nn.Module):
    def __init__(self, in_ch: int = 1, base: int = 32):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.bot = ConvBlock(base * 2, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)

        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        b = self.bot(p2)

        u2 = self.up2(b)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        z = self.out(d1)
        return z

# ============================================================
# Pre/post utilities (UNCHANGED)
# ============================================================

def mad(x: np.ndarray, eps: float = 1e-6) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med)) + eps)


def robust_normalize(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    img = img - np.median(img)
    img = img / (mad(img) + 1e-6)
    return img


def spatial_softmax_2d_logits(z: torch.Tensor, beta: float) -> torch.Tensor:
    """
    z: [B,1,H,W] logits -> p: [B,1,H,W] prob map (sum-to-1 per sample)
    p = exp(beta*(z-max))/sum(exp(beta*(z-max)))
    beta (β) controls sharpness: larger β => peakier distribution.
    """
    B, C, H, W = z.shape
    z2 = z.view(B, -1)
    z2 = z2 - z2.max(dim=1, keepdim=True).values
    p = torch.exp(beta * z2)
    p = p / (p.sum(dim=1, keepdim=True) + 1e-8)
    return p.view(B, 1, H, W)


def expected_xy_from_prob(p: torch.Tensor) -> torch.Tensor:
    """
    p: [B,1,H,W], sum-to-1
    return: [B,2] in 1-based (x=col, y=row)
    """
    B, _, H, W = p.shape
    device = p.device
    xs = torch.arange(1, W + 1, device=device, dtype=p.dtype).view(1, 1, 1, W)
    ys = torch.arange(1, H + 1, device=device, dtype=p.dtype).view(1, 1, H, 1)
    x = (p * xs).sum(dim=(1, 2, 3))
    y = (p * ys).sum(dim=(1, 2, 3))
    return torch.stack([x, y], dim=1)


def hard_argmax_xy(p: torch.Tensor) -> torch.Tensor:
    """
    p: [B,1,H,W]
    return: [B,2] 1-based (x=col, y=row)
    """
    B, _, H, W = p.shape
    idx = p.view(B, -1).argmax(dim=1)
    r = (idx // W) + 1
    c = (idx % W) + 1
    return torch.stack([c, r], dim=1).float()


def to01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    return np.clip(x, 0, 1)


def overlay_heatmap(gray01: np.ndarray, heat01: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """
    gray01: HxW in [0,1]
    heat01: HxW in [0,1]
    Returns RGB uint8 (heat injected into red channel for visibility)
    """
    base = np.stack([gray01, gray01, gray01], axis=2)
    out = base.copy()
    out[..., 0] = np.clip((1 - alpha) * out[..., 0] + alpha * heat01, 0, 1)
    return (out * 255).astype(np.uint8)


def draw_cross_rgb(img_rgb_u8: np.ndarray, x: float, y: float, color_rgb, r: int = 6, t: int = 2) -> np.ndarray:
    H, W, _ = img_rgb_u8.shape
    cx = int(round(x - 1))
    cy = int(round(y - 1))
    cx = max(0, min(W - 1, cx))
    cy = max(0, min(H - 1, cy))

    im = Image.fromarray(img_rgb_u8)
    dr = ImageDraw.Draw(im)
    for k in range(-(t // 2), (t // 2) + 1):
        dr.line([(cx - r, cy + k), (cx + r, cy + k)], fill=tuple(color_rgb), width=1)
        dr.line([(cx + k, cy - r), (cx + k, cy + r)], fill=tuple(color_rgb), width=1)
    return np.array(im, dtype=np.uint8)


def draw_roi_box_rgb(img_rgb_u8: np.ndarray, x: float, y: float, box_w: int, box_h: int, color_rgb, t: int = 2) -> np.ndarray:
    H, W, _ = img_rgb_u8.shape
    cx = int(round(x - 1))
    cy = int(round(y - 1))

    half_w = int(round(box_w / 2))
    half_h = int(round(box_h / 2))

    x0 = max(0, min(W - 1, cx - half_w))
    x1 = max(0, min(W - 1, cx + half_w))
    y0 = max(0, min(H - 1, cy - half_h))
    y1 = max(0, min(H - 1, cy + half_h))

    im = Image.fromarray(img_rgb_u8)
    dr = ImageDraw.Draw(im)
    for k in range(t):
        dr.rectangle([x0 - k, y0 - k, x1 + k, y1 + k], outline=tuple(color_rgb))
    return np.array(im, dtype=np.uint8)


def canon_to_raw(xy_canon, orig_h: int, orig_w: int, canon_h: int, canon_w: int):
    """Map 1-based canonical coords -> 1-based original coords"""
    x_c, y_c = float(xy_canon[0]), float(xy_canon[1])
    x_raw = x_c * (orig_w / float(canon_w))
    y_raw = y_c * (orig_h / float(canon_h))
    x_raw = float(np.clip(x_raw, 1.0, float(orig_w)))
    y_raw = float(np.clip(y_raw, 1.0, float(orig_h)))
    return x_raw, y_raw

# ============================================================
# Input loading (UNCHANGED)
# ============================================================

def load_user_input(uploaded_file):
    """
    Supports:
      - .mat containing 'image' (HxW or HxWxT)
      - standard image files -> grayscale
    Optionally reads GT (if present):
      - centroid_corrected else centroid
    Returns:
      dict with img2d (float32), orig_h, orig_w, gt_xy_raw, note, n_frames
    """
    name = uploaded_file.name.lower()
    out = {"gt_xy_raw": None, "note": "", "n_frames": 1}

    if name.endswith(".mat"):
        S = sio.loadmat(uploaded_file)
        if "image" not in S:
            raise ValueError("MAT file must contain variable `image`.")
        img = S["image"]

        if img.ndim == 2:
            out["n_frames"] = 1
        elif img.ndim == 3:
            out["n_frames"] = int(img.shape[2])
        else:
            raise ValueError(f"`image` must be HxW or HxWxT, got shape={img.shape}")

        gt = None
        if "centroid_corrected" in S and S["centroid_corrected"] is not None:
            c = np.array(S["centroid_corrected"]).squeeze().astype(np.float32)
            if c.size >= 2:
                gt = (float(c[0]), float(c[1]))
        if gt is None and "centroid" in S and S["centroid"] is not None:
            c = np.array(S["centroid"]).squeeze().astype(np.float32)
            if c.size >= 2:
                gt = (float(c[0]), float(c[1]))

        out["mat_image"] = img.astype(np.float32)
        out["gt_xy_raw"] = gt
        out["note"] = f"Loaded MAT file with image shape {img.shape}."

        if img.ndim == 2:
            img2d = img
        else:
            img2d = img[:, :, 0]
            out["note"] += " Default preview uses frame 0."

        img2d = img2d.astype(np.float32)
        out["orig_h"], out["orig_w"] = int(img2d.shape[0]), int(img2d.shape[1])
        out["img2d"] = img2d
        return out

    im = Image.open(uploaded_file).convert("L")
    img2d = np.array(im).astype(np.float32)
    out["orig_h"], out["orig_w"] = int(img2d.shape[0]), int(img2d.shape[1])
    out["img2d"] = img2d
    out["note"] = f"Loaded image file ({out['orig_h']}x{out['orig_w']}) converted to grayscale."
    return out

# ============================================================
# Checkpoint handling
# ============================================================

APP_DIR = Path(__file__).resolve().parent
DEFAULT_CKPT_NAME = "best_centroid_xy_net_DSNT_EPE.pt"
DEFAULT_CKPT_PATH = APP_DIR / DEFAULT_CKPT_NAME
DEMO_IMAGE_CANDIDATES = [
    APP_DIR / "demo_example.png",
    APP_DIR / "demo.png",
    APP_DIR / "high_cine.png",
]


def resolve_ckpt_path(user_text: str) -> Path:
    p = Path(user_text.strip())
    if str(p) == "":
        return DEFAULT_CKPT_PATH
    if p.is_absolute():
        return p
    return APP_DIR / p


def find_demo_image() -> Path | None:
    for p in DEMO_IMAGE_CANDIDATES:
        if p.exists():
            return p
    return None


def load_demo_input():
    demo_path = find_demo_image()
    if demo_path is None:
        return None
    im = Image.open(demo_path).convert("L")
    img2d = np.array(im).astype(np.float32)
    return {
        "img2d": img2d,
        "orig_h": int(img2d.shape[0]),
        "orig_w": int(img2d.shape[1]),
        "gt_xy_raw": None,
        "note": f"Showing built-in demo example: {demo_path.name}.",
        "n_frames": 1,
    }

# ============================================================
# Model loading/inference (UNCHANGED)
# ============================================================

@st.cache_resource
def load_model(ckpt_path_str: str, device: str):
    ckpt_path = str(ckpt_path_str)
    ckpt = torch.load(ckpt_path, map_location=device)

    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError("Checkpoint format unexpected. Expected a dict with key `'model'`.")

    model = TinyUNetLogits(in_ch=1, base=32).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, ckpt


def run_inference(
    model,
    img2d: np.ndarray,
    canon_h=256,
    canon_w=256,
    beta=40.0,
    device="cpu",
    compute_peak=False,
    capture_features=False,
):
    img_r = np.array(
        Image.fromarray(img2d.astype(np.float32)).resize((canon_w, canon_h), resample=Image.BILINEAR),
        dtype=np.float32,
    )

    img_rn = robust_normalize(img_r)
    X = torch.from_numpy(img_rn).unsqueeze(0).unsqueeze(0).to(device)

    hook = None
    feats = None
    if capture_features:
        hook = FeatureHook(model, layer_names=["enc1.net", "enc2.net", "bot.net", "dec1.net", "out"])

    with torch.no_grad():
        z = model(X)
        p = spatial_softmax_2d_logits(z, beta)

        xy = expected_xy_from_prob(p)[0].cpu().numpy()
        pk = hard_argmax_xy(p)[0].cpu().numpy() if compute_peak else None

        if capture_features and hook is not None:
            feats = {k: v.cpu() for k, v in hook.feats.items()}
            hook.close()

    z2d = z[0, 0].cpu().numpy().astype(np.float32)
    p2d = p[0, 0].cpu().numpy().astype(np.float32)

    xy_dsnt = (float(xy[0]), float(xy[1]))
    xy_peak = None if pk is None else (float(pk[0]), float(pk[1]))

    return img_r, z2d, p2d, xy_dsnt, xy_peak, feats

# ============================================================
# UI-only helper functions (NEW)
# ============================================================

def add_title_bar(img_rgb_u8: np.ndarray, title: str, bar_h: int = 30) -> np.ndarray:
    H, W, _ = img_rgb_u8.shape
    canvas = np.ones((H + bar_h, W, 3), dtype=np.uint8) * 255
    canvas[bar_h:, :, :] = img_rgb_u8
    im = Image.fromarray(canvas)
    dr = ImageDraw.Draw(im)
    dr.text((8, 7), title, fill=(0, 0, 0))
    return np.array(im, dtype=np.uint8)


def pad_to_same_height(imgs):
    hmax = max(im.shape[0] for im in imgs)
    out = []
    for im in imgs:
        H, W, C = im.shape
        if H < hmax:
            pad = np.ones((hmax - H, W, C), dtype=np.uint8) * 255
            im = np.concatenate([im, pad], axis=0)
        out.append(im)
    return out


def image_to_png_bytes(img_rgb_u8: np.ndarray) -> bytes:
    bio = BytesIO()
    Image.fromarray(img_rgb_u8).save(bio, format="PNG")
    return bio.getvalue()


def center_crop_nonblack(img_rgb_u8: np.ndarray, pad: int = 24) -> np.ndarray:
    gray = img_rgb_u8.mean(axis=2)
    mask = gray > 8
    if not np.any(mask):
        return img_rgb_u8
    ys, xs = np.where(mask)
    y0 = max(0, ys.min() - pad)
    y1 = min(img_rgb_u8.shape[0], ys.max() + pad + 1)
    x0 = max(0, xs.min() - pad)
    x1 = min(img_rgb_u8.shape[1], xs.max() + pad + 1)
    return img_rgb_u8[y0:y1, x0:x1, :]


def resize_rgb(img_rgb_u8: np.ndarray, target_w: int) -> np.ndarray:
    H, W, _ = img_rgb_u8.shape
    if W <= 0:
        return img_rgb_u8
    scale = target_w / float(W)
    target_h = max(1, int(round(H * scale)))
    return np.array(Image.fromarray(img_rgb_u8).resize((target_w, target_h), resample=Image.BILINEAR))


def make_triptych_figure(raw_rgb, canon_overlay, prob_rgb):
    a = add_title_bar(raw_rgb, "Detected Landmark on Original MRI")
    b = add_title_bar(canon_overlay, "Probability Overlay in Model Space")
    c = add_title_bar(prob_rgb, "Spatial Probability Map")
    a, b, c = pad_to_same_height([a, b, c])
    gap = 16
    spacer = np.ones((a.shape[0], gap, 3), dtype=np.uint8) * 255
    return np.concatenate([a, spacer, b, spacer.copy(), c], axis=1)


def make_side_by_side_figure(raw_rgb, canon_overlay):
    a = add_title_bar(raw_rgb, "Original MRI + Landmark")
    b = add_title_bar(canon_overlay, "Model-Space Overlay")
    a, b = pad_to_same_height([a, b])
    spacer = np.ones((a.shape[0], 16, 3), dtype=np.uint8) * 255
    return np.concatenate([a, spacer, b], axis=1)


def render_feature_maps(feats):
    st.divider()
    st.subheader("What did the network learn? (feature maps)")
    st.write(
        "Each block outputs multiple channels (feature maps). Early layers often highlight edges and contrast, "
        "while deeper layers become more task-specific and suppress background."
    )

    def topk_channels(feat_t: torch.Tensor, k=8):
        f = feat_t[0]
        scores = f.abs().mean(dim=(1, 2))
        idx = torch.topk(scores, k=min(k, f.shape[0])).indices.tolist()
        return idx

    def tensor_to_u8(t: torch.Tensor):
        x = t.numpy().astype(np.float32)
        x = x - x.min()
        x = x / (x.max() + 1e-8)
        return (x * 255).clip(0, 255).astype(np.uint8)

    for lname in ["enc1.net", "enc2.net", "bot.net", "dec1.net"]:
        if lname not in feats:
            continue
        ft = feats[lname]
        idxs = topk_channels(ft, k=8)
        st.markdown(f"**Layer: `{lname}`**")
        cols = st.columns(4)
        for i, ch in enumerate(idxs):
            fmap = ft[0, ch]
            img_u8 = tensor_to_u8(fmap)
            cols[i % 4].image(img_u8, caption=f"ch {ch}", use_container_width=True)

# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title="Interactive MRI Landmark Localization Teaching App", layout="wide")

st.title("Interactive MRI Landmark Localization Teaching App")
st.write(
    "This app demonstrates how a neural network localizes a target region in full-field-of-view MRI using a "
    "probability-based spatial representation. The layout below is optimized for clean screenshots and printing."
)

with st.sidebar:
    st.header("Settings")

    ckpt_text = st.text_input(
        "Checkpoint path (.pt) [relative ok]",
        value=DEFAULT_CKPT_NAME if DEFAULT_CKPT_PATH.exists() else "",
        help="By default, the app searches for the checkpoint in the same folder as this app.",
    )

    canon_h = st.number_input("Canonical height", value=256, min_value=64, max_value=1024, step=16)
    canon_w = st.number_input("Canonical width", value=256, min_value=64, max_value=1024, step=16)

    beta = st.number_input(
        "Softmax beta (β)",
        value=40.0,
        min_value=1.0,
        max_value=200.0,
        step=1.0,
        help="β controls the sharpness of the probability map, not a threshold.",
    )
    alpha = st.slider("Heatmap overlay alpha", min_value=0.0, max_value=1.0, value=0.35, step=0.05)

    st.subheader("Toggles")
    show_peak = st.checkbox("Show highest-probability point (argmax)", value=False)
    show_net = st.checkbox("Show network structure", value=False)
    show_beta = st.checkbox("Show β (beta) explanation", value=False)
    show_torchinfo = st.checkbox("Show layer-by-layer summary (torchinfo)", value=False)
    show_features = st.checkbox("Show intermediate feature maps", value=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.caption(f"Device: {device}")

# ------------------------------------------------------------
# Top area: uploader + preview
# ------------------------------------------------------------

top_left, top_right = st.columns([0.95, 1.25])

with top_left:
    st.markdown("### Upload an image or .mat")
    uploaded = st.file_uploader(
        "Upload an image or .mat",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "mat"],
        label_visibility="collapsed",
    )
    st.caption("PNG, JPG, JPEG, TIF, TIFF, BMP, or MATLAB .mat")

if uploaded is not None:
    try:
        data = load_user_input(uploaded)
        input_mode = "upload"
    except Exception as e:
        st.error("Failed to read file.")
        st.exception(e)
        st.stop()
else:
    data = load_demo_input()
    input_mode = "demo"
    if data is None:
        with top_right:
            st.markdown("### Example preview")
            st.info("Place `demo_example.png` in the same folder as this app to show a built-in example here.")
        st.stop()

img2d = data["img2d"]
orig_h, orig_w = int(data["orig_h"]), int(data["orig_w"])
gt_xy_raw = data.get("gt_xy_raw", None)

if "mat_image" in data and data["mat_image"].ndim == 3:
    with top_left:
        T = data["n_frames"]
        frame_idx = st.slider("Select frame (MAT only)", min_value=0, max_value=T - 1, value=0, step=1)
    img2d = data["mat_image"][:, :, frame_idx].astype(np.float32)
    orig_h, orig_w = int(img2d.shape[0]), int(img2d.shape[1])

preview01 = to01(img2d)
preview_rgb = (np.stack([preview01, preview01, preview01], axis=2) * 255).astype(np.uint8)

with top_right:
    st.markdown("### Example preview")
    with st.container(border=True):
        st.image(
            preview_rgb,
            use_container_width=True,
            caption="Built-in demonstration example." if input_mode == "demo" else "Uploaded input preview.",
        )
    st.caption(data["note"] if input_mode == "demo" else "Uploaded image loaded successfully.")

st.divider()

# ------------------------------------------------------------
# Resolve checkpoint
# ------------------------------------------------------------

ckpt_path = resolve_ckpt_path(ckpt_text if ckpt_text.strip() else DEFAULT_CKPT_NAME)
if not ckpt_path.exists():
    st.error(
        f"Checkpoint not found: {ckpt_path}\n\n"
        f"Put `{DEFAULT_CKPT_NAME}` in the same folder as this app, or input a valid path."
    )
    st.stop()

# ------------------------------------------------------------
# Run model + prepare visuals
# ------------------------------------------------------------

try:
    model, _ckpt = load_model(str(ckpt_path), device=device)

    img_canon, z2d, p2d, xy_dsnt, xy_peak, feats = run_inference(
        model,
        img2d,
        canon_h=int(canon_h),
        canon_w=int(canon_w),
        beta=float(beta),
        device=device,
        compute_peak=bool(show_peak),
        capture_features=bool(show_features),
    )

    xy_dsnt_raw = canon_to_raw(xy_dsnt, orig_h, orig_w, canon_h=int(canon_h), canon_w=int(canon_w))
    xy_peak_raw = None
    if xy_peak is not None:
        xy_peak_raw = canon_to_raw(xy_peak, orig_h, orig_w, canon_h=int(canon_h), canon_w=int(canon_w))

    roi_w_canon = int(canon_w) // 2
    roi_h_canon = int(canon_h) // 2
    roi_w_raw = orig_w // 2
    roi_h_raw = orig_h // 2

    gray01 = to01(img_canon)
    heat01 = to01(p2d)

    canon_overlay = overlay_heatmap(gray01, heat01, alpha=float(alpha))
    canon_overlay = draw_cross_rgb(canon_overlay, xy_dsnt[0], xy_dsnt[1], color_rgb=(255, 0, 0))
    canon_overlay = draw_roi_box_rgb(canon_overlay, xy_dsnt[0], xy_dsnt[1], roi_w_canon, roi_h_canon, color_rgb=(0, 255, 255), t=2)
    if xy_peak is not None:
        canon_overlay = draw_cross_rgb(canon_overlay, xy_peak[0], xy_peak[1], color_rgb=(255, 255, 0))

    prob_u8 = (heat01 * 255.0).clip(0, 255).astype(np.uint8)
    prob_rgb = np.stack([prob_u8, prob_u8, prob_u8], axis=2)

    raw01 = to01(img2d)
    raw_rgb = (np.stack([raw01, raw01, raw01], axis=2) * 255).astype(np.uint8)
    raw_rgb = draw_cross_rgb(raw_rgb, xy_dsnt_raw[0], xy_dsnt_raw[1], color_rgb=(255, 0, 0))
    raw_rgb = draw_roi_box_rgb(raw_rgb, xy_dsnt_raw[0], xy_dsnt_raw[1], roi_w_raw, roi_h_raw, color_rgb=(0, 255, 255), t=2)
    if xy_peak_raw is not None:
        raw_rgb = draw_cross_rgb(raw_rgb, xy_peak_raw[0], xy_peak_raw[1], color_rgb=(255, 255, 0))
    if gt_xy_raw is not None:
        raw_rgb = draw_cross_rgb(raw_rgb, gt_xy_raw[0], gt_xy_raw[1], color_rgb=(0, 255, 0))

    canon_overlay_crop = center_crop_nonblack(canon_overlay, pad=32)
    prob_rgb_crop = center_crop_nonblack(prob_rgb, pad=32)
    canon_overlay_panel = resize_rgb(canon_overlay_crop, target_w=520)
    prob_rgb_panel = resize_rgb(prob_rgb_crop, target_w=520)

    nsf_triptych = make_triptych_figure(raw_rgb, canon_overlay_panel, prob_rgb_panel)
    nsf_side_by_side = make_side_by_side_figure(raw_rgb, canon_overlay_panel)

except Exception as e:
    st.error("Model inference failed.")
    st.exception(e)
    st.stop()

# ============================================================
# Main result first: avoid the giant black panel
# ============================================================

st.subheader("Main Detection Result")

main_left, main_right = st.columns([1.05, 1.0])
with main_left:
    st.image(
        raw_rgb,
        caption="Original MRI with detected landmark and automatically scaled ROI.",
        use_container_width=True,
    )
with main_right:
    st.image(
        nsf_side_by_side,
        caption="Compact summary figure for screenshots and printing.",
        use_container_width=True,
    )

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
with metric_col1:
    st.metric("x (original)", f"{xy_dsnt_raw[0]:.1f}")
with metric_col2:
    st.metric("y (original)", f"{xy_dsnt_raw[1]:.1f}")
with metric_col3:
    st.metric("ROI width", f"{roi_w_raw}")
with metric_col4:
    st.metric("ROI height", f"{roi_h_raw}")

# ============================================================
# Downloadable figure outputs
# ============================================================

st.divider()
st.subheader("Downloadable Figure Outputs")

fig_col1, fig_col2 = st.columns([1, 1])
with fig_col1:
    st.image(nsf_side_by_side, caption="Compact figure for slides or screenshots.", use_container_width=True)
    st.download_button(
        label="Download Compact Figure (PNG)",
        data=image_to_png_bytes(nsf_side_by_side),
        file_name="mri_landmark_localization_compact.png",
        mime="image/png",
    )

with fig_col2:
    st.image(nsf_triptych, caption="Three-panel figure including cropped model-space panels.", use_container_width=True)
    st.download_button(
        label="Download Three-Panel Figure (PNG)",
        data=image_to_png_bytes(nsf_triptych),
        file_name="mri_landmark_localization_triptych.png",
        mime="image/png",
    )

# ============================================================
# Supporting panels below the primary figure
# ============================================================

st.divider()
st.subheader("Supporting Visualizations")

supp1, supp2 = st.columns([1, 1])
with supp1:
    st.image(canon_overlay_panel, caption="Cropped model-space probability overlay.", use_container_width=True)
with supp2:
    st.image(prob_rgb_panel, caption="Cropped spatial probability map.", use_container_width=True)

st.divider()
st.subheader("Predicted Coordinates and Quantitative Summary")

sum1, sum2 = st.columns([1, 1])
with sum1:
    st.write("**DSNT centroid (canonical x,y):**")
    st.code(f"({xy_dsnt[0]:.2f}, {xy_dsnt[1]:.2f})")

    st.write("**DSNT centroid (original x,y):**")
    st.code(f"({xy_dsnt_raw[0]:.2f}, {xy_dsnt_raw[1]:.2f})")

    if xy_peak is not None:
        st.write("**Argmax / peak (canonical x,y):**")
        st.code(f"({xy_peak[0]:.2f}, {xy_peak[1]:.2f})")

    if xy_peak_raw is not None:
        st.write("**Argmax / peak (original x,y):**")
        st.code(f"({xy_peak_raw[0]:.2f}, {xy_peak_raw[1]:.2f})")

with sum2:
    st.write("**ROI size (canonical):**")
    st.code(f"{roi_w_canon} x {roi_h_canon}")

    st.write("**ROI size (original):**")
    st.code(f"{roi_w_raw} x {roi_h_raw}")

    if gt_xy_raw is not None:
        dx = xy_dsnt_raw[0] - gt_xy_raw[0]
        dy = xy_dsnt_raw[1] - gt_xy_raw[1]
        epe = float(np.sqrt(dx * dx + dy * dy))
        st.write("**Ground truth centroid (original x,y):**")
        st.code(f"({gt_xy_raw[0]:.2f}, {gt_xy_raw[1]:.2f})")
        st.write("**EPE (DSNT vs GT) in original pixels:**")
        st.code(f"{epe:.2f} px")

# ============================================================
# Optional teaching sections last
# ============================================================

if show_net:
    with st.expander("Network structure (TinyUNetLogits)", expanded=False):
        st.code(str(TinyUNetLogits(in_ch=1, base=32)), language="text")

if show_torchinfo:
    with st.expander("Layer-by-layer summary (optional: torchinfo)", expanded=False):
        try:
            from torchinfo import summary
            m = TinyUNetLogits(in_ch=1, base=32)
            s = summary(m, input_size=(1, 1, int(canon_h), int(canon_w)), verbose=0)
            st.text(str(s))
        except Exception as e:
            st.warning("torchinfo is not available or failed. Install with: pip install torchinfo")
            st.exception(e)

if show_beta:
    with st.expander("How β and DSNT work (simple explanation)"):
        st.code(
            "β is NOT a threshold.\n\n"
            "The network outputs logits z(x,y).\n\n"
            "Spatial softmax:\n"
            "p(x,y) = exp(β * z(x,y)) / sum_{u,v} exp(β * z(u,v))\n\n"
            "Effect of β:\n"
            "- Larger β  -> sharper probability map\n"
            "- Smaller β -> flatter probability map\n\n"
            "DSNT centroid (red cross):\n"
            "x_hat = sum_{x,y} p(x,y) * x\n"
            "y_hat = sum_{x,y} p(x,y) * y\n\n"
            "Argmax (yellow cross):\n"
            "Argmax = location of max p(x,y)\n\n"
            "No threshold is used. β only controls concentration."
        )

with st.expander("How the network turns features into a probability map"):
    st.write("Logits are the raw scores from the last convolution. Spatial softmax transforms them into probabilities.")
    pcol1, pcol2 = st.columns([1, 1])
    with pcol1:
        st.image(
            np.stack([(to01(z2d) * 255).astype(np.uint8)] * 3, axis=2),
            caption="Logits",
            use_container_width=True,
        )
    with pcol2:
        st.image(
            prob_rgb_panel,
            caption="Cropped probability map after spatial softmax",
            use_container_width=True,
        )

if show_features and feats is not None:
    render_feature_maps(feats)
