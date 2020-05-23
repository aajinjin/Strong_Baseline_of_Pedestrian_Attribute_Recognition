from functools import partial

from models.base_block import BaseClassifier

from models.resnet import resnet50, resnet101, resnext50_32x4d

from models.senet import se_resnet50

base = partial(BaseClassifier)

resnet50 = partial(resnet50, pretrained=True)
resnet101 = partial(resnet101, pretrained=True)
resnext50 = partial(resnext50_32x4d, pretrained=True)


senet50 = partial(se_resnet50)

