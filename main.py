import torch
import main
import torchvision.transforms.functional
from torchvision import transforms
from PIL import Image
import numpy as np

def halide(tensor_in):
    tensor_out = torch.empty_like(tensor_in)
    # main.adjust_brightness(tensor_in, 1000, tensor_out)
    main.adjust_hue(tensor_in, 0.1, tensor_out)
    return tensor_out


def pytorch(tensor_in):
    return torchvision.transforms.functional.adjust_hue(tensor_in, 0.1)


# Open image with pillow and create a transform to convert it to a PyTorch tensor
img = Image.open("Images/dog1.jpg")
convert = transforms.ToTensor()
tensor = convert(img)

tensor2 = torch.ones(3, 2000, 2000)

# Get output tensor from both functions
pytorch_output = pytorch(tensor)  # For some reason, calling pytorch after halide messes up the halide_output
halide_output = halide(tensor)
# print(pytorch_output)
# print(halide_output)

# Test if they are the same
print(torch.allclose(halide_output, pytorch_output))

# Save image if you want to take a look
print(np.array(img.convert('HSV'))[0])
print("Halide: ")
pil = transforms.functional.to_pil_image(halide_output)
pil.save("Images/halide_output.jpg")
print(np.array(pil.convert('HSV'))[0])

print('Pytorch: ')
pil = transforms.functional.to_pil_image(pytorch_output)
pil.save("Images/pytorch_output.jpg")
print(np.array(pil.convert('HSV'))[0])