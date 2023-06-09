from torchvision import transforms

def get_transforms(input_size):
    image_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Pad((1, 0)),
                                          transforms.Resize(input_size),])
    mask_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Pad((1, 0)),
                                          transforms.Resize(input_size),])
    test_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Pad((1, 0)),
                                          transforms.Resize(input_size),])
    return {'image': image_transform, 'mask': mask_transform, 'test': test_transform}