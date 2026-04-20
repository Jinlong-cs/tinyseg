import types

import torch


def attention_forward(self, x):
    b, c, h, w = x.shape
    n = h * w
    qkv = self.qkv(x)
    q, k, v = qkv.view(b, self.num_heads, self.key_dim * 2 + self.head_dim, n).split(
        [self.key_dim, self.key_dim, self.head_dim], dim=2
    )
    attn = (q.transpose(-2, -1) @ k) * self.scale
    attn = attn.permute(0, 3, 1, 2).contiguous()
    max_attn = attn.max(dim=1, keepdim=True).values
    exp_attn = torch.exp(attn - max_attn)
    attn = exp_attn / exp_attn.sum(dim=1, keepdim=True)
    attn = attn.permute(0, 2, 3, 1).contiguous()
    x = (v @ attn.transpose(-2, -1)).view(b, c, h, w) + self.pe(v.reshape(b, c, h, w))
    return self.proj(x)


def aattn_forward(self, x):
    b, c, h, w = x.shape
    n = h * w
    qkv = self.qkv(x).flatten(2).transpose(1, 2)
    if self.area > 1:
        qkv = qkv.reshape(b * self.area, n // self.area, c * 3)
        b, n, _ = qkv.shape
    q, k, v = (
        qkv.view(b, n, self.num_heads, self.head_dim * 3)
        .permute(0, 2, 3, 1)
        .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
    )
    attn = (q.transpose(-2, -1) @ k) * (self.head_dim**-0.5)
    attn = attn.permute(0, 3, 1, 2).contiguous()
    max_attn = attn.max(dim=1, keepdim=True).values
    exp_attn = torch.exp(attn - max_attn)
    attn = exp_attn / exp_attn.sum(dim=1, keepdim=True)
    attn = attn.permute(0, 2, 3, 1).contiguous()
    x = v @ attn.transpose(-2, -1)
    x = x.permute(0, 3, 1, 2)
    v = v.permute(0, 3, 1, 2)
    if self.area > 1:
        x = x.reshape(b // self.area, n * self.area, c)
        v = v.reshape(b // self.area, n * self.area, c)
        b, n, _ = x.shape
    x = x.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
    v = v.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
    x = x + self.pe(v)
    return self.proj(x)


def classify_forward(self, x):
    x = torch.cat(x, 1) if isinstance(x, list) else x
    return self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))


def detect_forward(self, x):
    outputs = []
    for i in range(self.nl):
        outputs.append(self.cv3[i](x[i]).permute(0, 2, 3, 1).contiguous())
        outputs.append(self.cv2[i](x[i]).permute(0, 2, 3, 1).contiguous())
    return outputs


def v10_detect_forward(self, x):
    outputs = []
    for i in range(self.nl):
        outputs.append(self.one2one_cv3[i](x[i]).permute(0, 2, 3, 1).contiguous())
        outputs.append(self.one2one_cv2[i](x[i]).permute(0, 2, 3, 1).contiguous())
    return outputs


def segment_forward(self, x):
    outputs = []
    for i in range(self.nl):
        outputs.append(self.cv3[i](x[i]).permute(0, 2, 3, 1).contiguous())
        outputs.append(self.cv2[i](x[i]).permute(0, 2, 3, 1).contiguous())
        outputs.append(self.cv4[i](x[i]).permute(0, 2, 3, 1).contiguous())
    proto_input = x if hasattr(self.proto, "feat_refine") else x[0]
    proto = self.proto(proto_input)
    if isinstance(proto, tuple):
        proto = proto[0]
    outputs.append(proto.permute(0, 2, 3, 1).contiguous())
    return outputs


def pose_forward(self, x):
    outputs = []
    for i in range(self.nl):
        outputs.append(self.cv3[i](x[i]).permute(0, 2, 3, 1).contiguous())
        outputs.append(self.cv2[i](x[i]).permute(0, 2, 3, 1).contiguous())
        outputs.append(self.cv4[i](x[i]).permute(0, 2, 3, 1).contiguous())
    return outputs


def obb_forward(self, x):
    outputs = []
    for i in range(self.nl):
        outputs.append(self.cv3[i](x[i]).permute(0, 2, 3, 1).contiguous())
        outputs.append(self.cv2[i](x[i]).permute(0, 2, 3, 1).contiguous())
        outputs.append(self.cv4[i](x[i]).permute(0, 2, 3, 1).contiguous())
    return outputs


def patch_model_for_rdk(model):
    from ultralytics.nn.modules.block import AAttn, Attention
    from ultralytics.nn.modules.head import Classify, Detect, OBB, Pose, Segment, v10Detect

    for child in model.children():
        if isinstance(child, Classify):
            child.forward = types.MethodType(classify_forward, child)
        elif isinstance(child, Segment):
            child.forward = types.MethodType(segment_forward, child)
        elif isinstance(child, Pose):
            child.forward = types.MethodType(pose_forward, child)
        elif isinstance(child, OBB):
            child.forward = types.MethodType(obb_forward, child)
        elif isinstance(child, v10Detect):
            child.forward = types.MethodType(v10_detect_forward, child)
        elif isinstance(child, Detect):
            child.forward = types.MethodType(detect_forward, child)
        elif isinstance(child, AAttn):
            child.forward = types.MethodType(aattn_forward, child)
        elif isinstance(child, Attention):
            child.forward = types.MethodType(attention_forward, child)
        patch_model_for_rdk(child)
