"""The beetle model: crop(s) of beetle specimens -> SPB / Cleridae / other_Coleoptera."""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # BeetleID
BEETLE_MODEL = os.path.join(ROOT, "models", "beetle_classifier_resnet18.pth")
BEETLE_NAMES = os.path.join(ROOT, "models", "beetle_class_mapping.txt")


def _class_names(path, n):
    names = {}
    for line in open(path, encoding="utf-8"):
        m = re.match(r"Class:\s*(.+?),\s*Class id:\s*(\d+)", line.strip())
        if m:
            names[int(m.group(2))] = m.group(1)
    return [names.get(i, f"class {i}") for i in range(n)]


class BeetleClassifier:
    """Loads the beetle model once; classify() takes a list of RGB crops (numpy arrays)."""

    def __init__(self, device="cuda", weights=BEETLE_MODEL, mapping=BEETLE_NAMES):
        import timm
        import torch
        import torchvision.transforms as T
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        a = ckpt["args"]                      # training/classification/finetune.py saves its arguments
        self.model = timm.create_model(a.model, pretrained=False, num_classes=a.nb_classes)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()
        self.names = _class_names(mapping, a.nb_classes)
        self.transform = T.Compose([          # finetune.py's validation transform
            T.Resize(a.input_size, interpolation=3), T.CenterCrop(a.input_size), T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def classify(self, crops):
        """[RGB array] -> [{"class", "probability", "probabilities": {name: p}}]"""
        import torch
        from PIL import Image
        if not len(crops):
            return []
        with torch.no_grad():
            x = torch.stack([self.transform(Image.fromarray(c)) for c in crops]).to(self.device)
            probs = self.model(x).softmax(1).cpu()
        out = []
        for p in probs:
            k = int(p.argmax())
            out.append({"class": self.names[k], "probability": round(float(p[k]), 3),
                        "probabilities": {n: round(float(v), 3) for n, v in zip(self.names, p)}})
        return out
