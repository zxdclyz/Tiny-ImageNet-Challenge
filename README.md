# Tiny-ImageNet-Challenge

The first course project of Introduction to Deep Learning, hosted by Prof. Xiaolin Hu and TAs.

## Dataset

Download dataset here https://cloud.tsinghua.edu.cn/d/44ae22e82c274b968189/

## About Competition

### task

对于来自 ImageNet 的 100 个种类的图片，我们要对其进行分类

其中每一个种类有 1,000 training images，100 validation images，最终的测试集有 10000 张图片

> For this course project, you need to consider how to achieve high classification accuracy on both general ImageNet images and natural adversarial examples.

如上所述，我们的任务不只是进行分类，还要处理一些 Natural Adversarial Examples，这些“自然对抗”的图片一般是含有遮挡、天气以及其他场景因素的干扰，这里有相关的一篇论文[Natural Adversarial Examples](https://arxiv.org/pdf/1907.07174.pdf)

除此之外，我们拿到的数据集中的图片并不是 ImageNet 标准的 224x224，而是更小的 64x64，根据老师上课时提到的，这也是我们会遇到的一个困难

### Dataloader

由于数据集并不是标准的 ImageNet，我们需要自己修改 Dataloader，助教给出了例子如下

```python
# Tiny ImangeNet Dataloader
import numpy as np
from PIL import Image
def default_loader(path):
    return Image.open(path).convert('RGB')
class TinyImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, root, data_list, transform = None, loader = default_loader):
        # root: your_path/TinyImageNet/
        # data_list: your_path/TinyImageNet/train.txt etc.
        images = []
        labels = open(data_list).readlines()
        for line in labels:
            items = line.strip('\n').split()
            img_name = items[0]

            # test list contains only image name
            test_flag = True if len(items) == 1 else False
            label = None if test_flag == True else np.array(int(items[1]))

            if os.path.isfile(os.path.join(root, img_name)):
                images.append((img_name, label))
            else:
                print(os.path.join(root, img_name) + 'Not Found.')

        self.root = root
        self.images = images
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_name, label = self.images[index]
        img = self.loader(os.path.join(self.root, img_name))
        raw_img = img.copy()
        if self.transform is not None:
            img = self.transform(img)
        return (img, label) if label is not None else img

    def __len__(self):
        return len(self.images)
```

其中内容比较简单，在 init 的时候把 list（数据集中的 test.txt 和 train.txt）中的文件路径和它对应的 label（如果有，测试集是没有的）读到 dataloader 中，然后在 getitem 的时候用 PIL 把图像读出来然后返回，如果图像带 label，就返回一个元组。

目前对于这个 dataloader 如何使用还存疑，是否可以像 pytorch 里的 loader 一样使用？
