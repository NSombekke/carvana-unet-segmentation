from torchvision import transforms
from torch.nn.functional import pad

def get_transforms(input_size):
    image_transform = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Pad((1, 0)),
                                         transforms.Resize(input_size),
                                         transforms.ToTensor(),])
    mask_transform = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Pad((1, 0)),
                                         transforms.Resize(input_size),
                                         transforms.ToTensor(),])
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Pad((1, 0)),])
    return {'image': image_transform, 'mask': mask_transform, 'test': test_transform}