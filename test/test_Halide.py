import pytest
import torch
import main
import torchvision.io
from PIL import Image
import torchvision.transforms.functional
from torchvision import transforms
from pathlib import Path

# Feel free to change the images path if required
img1 = str(Path(__file__).parent / ".." / "Images" / "dog1.jpg" )
img2 = str(Path(__file__).parent / ".." / "Images" / "dog2.jpg" )

@pytest.mark.parametrize("path", [img1, img2])
@pytest.mark.parametrize("precision", [1.0, 1.2, 3.1415926])

def test_halide_brightness(path, precision):
    img = Image.open(path)
    convert = transforms.ToTensor()
    image_tensor = convert(img)
    halide_output = torch.clone(image_tensor)

    torchvision_output = torchvision.transforms.functional.adjust_brightness(image_tensor, precision)
    main.adjust_brightness(image_tensor, precision, halide_output)

    assert torch.allclose(torchvision_output, halide_output)

@pytest.mark.parametrize("path", [img1, img2])
def test_halide_grayscale(path):
    img = Image.open(path)
    convert = transforms.ToTensor()
    image_tensor = convert(img)
    halide_output = torch.clone(image_tensor)

    torchvision_output = torchvision.transforms.functional.rgb_to_grayscale(image_tensor)
    main.rgb_to_grayscale(image_tensor, halide_output)

    assert torch.allclose(torchvision_output, halide_output)

@pytest.mark.parametrize("path", [img1, img2])
def test_invert(path):
    img = Image.open(path)
    convert = transforms.ToTensor()
    image_tensor = convert(img)
    halide_output = torch.clone(image_tensor)

    torchvision_output = torchvision.transforms.functional.invert(image_tensor)
    main.invert(image_tensor, halide_output)

    assert torch.allclose(torchvision_output, halide_output)


@pytest.mark.parametrize("path", [img1, img2])
def test_hflip(path):
    img = Image.open(path)
    convert = transforms.ToTensor()
    image_tensor = convert(img)
    halide_output = torch.clone(image_tensor)

    torchvision_output = torchvision.transforms.functional.hflip(image_tensor)
    main.hflip(image_tensor, halide_output)

    assert torch.allclose(torchvision_output, halide_output)



@pytest.mark.parametrize("path", [img1, img2])
def test_vflip(path):
    img = Image.open(path)
    convert = transforms.ToTensor()
    image_tensor = convert(img)
    halide_output = torch.clone(image_tensor)

    torchvision_output = torchvision.transforms.functional.vflip(image_tensor)
    main.vflip(image_tensor, halide_output)

    assert torch.allclose(torchvision_output, halide_output)


@pytest.mark.parametrize("path", [img1, img2])
@pytest.mark.parametrize("i", [0])
@pytest.mark.parametrize("j", [0])
@pytest.mark.parametrize("h", [10])
@pytest.mark.parametrize("w", [10])
def test_erase_tensor(path, i, j, h, w):
    img = Image.open(path)
    convert = transforms.ToTensor()
    image_tensor = convert(img)
    halide_output = torch.clone(image_tensor)

    v = torch.randint(0, 1, (3, 10, 10), dtype=torch.float32)

    torchvision_output = torchvision.transforms.functional.erase(image_tensor,i,j,w,h,v)
    main.erase_tensor(image_tensor, i,j,w,h,v, halide_output)
    assert torch.allclose(torchvision_output, halide_output)



#TODO fix
@pytest.mark.parametrize("path", [img1, img2])
@pytest.mark.parametrize("i", [0])
@pytest.mark.parametrize("j", [0])
@pytest.mark.parametrize("h", [10])
@pytest.mark.parametrize("w", [10])
def test_crop(path, i, j, h, w):
    img = Image.open(path)
    convert = transforms.ToTensor()
    image_tensor = convert(img)
    halide_output = torch.clone(image_tensor)

    torchvision_output = torchvision.transforms.functional.crop(image_tensor,i,j,h,w)
    main.crop(image_tensor, i,j,h,w, halide_output)

    assert torch.allclose(torchvision_output, halide_output)


@pytest.mark.parametrize("path", [img1, img2])
@pytest.mark.parametrize("threshold", [0.1, 0.2, 1])
def test_solarize(path, threshold):
    img = Image.open(path)
    convert = transforms.ToTensor()
    image_tensor = convert(img)
    halide_output = torch.clone(image_tensor)

    torchvision_output = torchvision.transforms.functional.solarize(image_tensor, threshold)
    main.solarize(image_tensor, threshold, halide_output)

    assert torch.allclose(torchvision_output, halide_output)

@pytest.mark.parametrize("path", [img1, img2])
@pytest.mark.parametrize("saturation", [1, 1.5, 2])
def test_halide_adjust_saturation(path, saturation):
    img = Image.open(path)
    convert = transforms.ToTensor()
    image_tensor = convert(img)

    torchvision_output = torchvision.transforms.functional.adjust_saturation(image_tensor, saturation)

    halide_output = main.adjust_saturation(image_tensor, saturation)

    assert torch.allclose(torchvision_output, halide_output)

@pytest.mark.parametrize("path", [img1, img2])
@pytest.mark.parametrize("gamma", [0.3, 0.7, 1, 1.5, 3])
@pytest.mark.parametrize("gain", [1, 2])
def test_halide_adjust_gamma(path, gamma, gain):
    img = Image.open(path)
    convert = transforms.ToTensor()
    image_tensor = convert(img)

    torchvision_output = torchvision.transforms.functional.adjust_gamma(image_tensor, gamma, gain)

    halide_output = main.adjust_gamma(image_tensor, gamma, gain)

    assert torch.allclose(torchvision_output, halide_output)

@pytest.mark.parametrize("path", [img1, img2])
@pytest.mark.parametrize("hue_factor", [-0.5, -0.2, 0, 0.1, 0.5])
def test_halide_adjust_hue(path, hue_factor):
    img = Image.open(path)
    convert = transforms.ToTensor()
    image_tensor = convert(img)

    torchvision_output = torchvision.transforms.functional.adjust_hue(image_tensor, hue_factor)

    halide_output = main.adjust_hue(image_tensor, hue_factor)

    assert torch.allclose(torchvision_output, halide_output)

@pytest.mark.parametrize("path", [img1, img2])
@pytest.mark.parametrize("sharpness_factor", [0, 0.5, 1, 1.5, 2])
def test_halide_adjust_sharpness(path, sharpness_factor):
    img = Image.open(path)
    convert = transforms.ToTensor()
    image_tensor = convert(img)

    torchvision_output = torchvision.transforms.functional.adjust_sharpness(image_tensor, sharpness_factor)

    halide_output = main.adjust_sharpness(image_tensor, sharpness_factor)

    assert torch.allclose(torchvision_output, halide_output)


@pytest.mark.parametrize("path", [img1, img2])
@pytest.mark.parametrize("startpoints", [[50, 50], [100, 50], [100, 100], [50, 100]])
@pytest.mark.parametrize("endpoints", [[100, 200], [200, 200], [200, 300], [150, 300]])
def test_halide_perspective(path, startpoints, endpoints):
    img = Image.open(path)
    convert = transforms.ToTensor()
    image_tensor = convert(img)

    torchvision_output = torchvision.transforms.functional.perspective(image_tensor, startpoints, endpoints)

    halide_output = main.perspective(image_tensor, startpoints, endpoints)

    assert torch.allclose(torchvision_output, halide_output)