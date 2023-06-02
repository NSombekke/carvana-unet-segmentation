from torchvision import transforms

def get_transforms(image_size):
    image_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Pad((1, 0)),])
    mask_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Pad((1, 0)),])
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Pad((1, 0)),])
    return {'image': image_transform, 'mask': mask_transform, 'test': test_transform}