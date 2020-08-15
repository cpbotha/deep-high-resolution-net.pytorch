"""cell-based python execution version of DSNT basic usage guide:
https://github.com/anibali/dsntnn/blob/master/examples/basic_usage.md

Best is to execute these cells in Visual Studio Code or PyCharm Professional, or
anything else with good scipy and cell support.

-- cpbotha
"""

# %% imports
import dsntnn
import torch
from torch import nn
from torch import optim
# cpbotha: let's throw in mixed precision while we're at it thanks pytorch!
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import scipy.misc
import skimage.transform


# %% define networks and modules
class FCN(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.layers(x)


class CoordRegressionNetwork(nn.Module):
    def __init__(self, n_locations):
        super().__init__()
        self.fcn = FCN()
        self.hm_conv = nn.Conv2d(16, n_locations, kernel_size=1, bias=False)

    def forward(self, images):
        # 1. Run the images through our FCN
        fcn_out = self.fcn(images)
        # 2. Use a 1x1 conv to get one unnormalized heatmap per location
        unnormalized_heatmaps = self.hm_conv(fcn_out)
        # 3. Normalize the heatmaps
        heatmaps = dsntnn.flat_softmax(unnormalized_heatmaps)
        # 4. Calculate the coordinates
        coords = dsntnn.dsnt(heatmaps)

        return coords, heatmaps


# %% look raccoon in the eye
image_size = [40, 40]

# consider making PR for https://github.com/anibali/dsntnn/blob/master/examples/basic_usage.md#training-the-model
# it's still using scipy.misc.imresize
raccoon_face = skimage.transform.resize(scipy.misc.face()[200:400, 600:800, :], image_size)
eye_x, eye_y = 24, 26

plt.imshow(raccoon_face)
plt.scatter([eye_x], [eye_y], color='red', marker='X')
plt.show()

# %%

raccoon_face_tensor = torch.from_numpy(raccoon_face).permute(2, 0, 1).float()
input_tensor = raccoon_face_tensor.div(255).unsqueeze(0)
input_var = input_tensor.cuda()

eye_coords_tensor = torch.Tensor([[[eye_x, eye_y]]])
target_tensor = (eye_coords_tensor * 2 + 1) / torch.Tensor(image_size) - 1
target_var = target_tensor.cuda()

print('Target: {:0.4f}, {:0.4f}'.format(*list(target_tensor.squeeze())))

# %% setup the network

model = CoordRegressionNetwork(n_locations=1).cuda()

coords, heatmaps = model(input_var)

print('Initial prediction: {:0.4f}, {:0.4f}'.format(*list(coords[0, 0])))
plt.imshow(heatmaps[0, 0].detach().cpu().numpy())
plt.show()

# %% now train the network
# amp scaler
scaler = GradScaler()

optimizer = optim.RMSprop(model.parameters(), lr=2.5e-4)

for i in range(200):
    # clear gradients
    optimizer.zero_grad()

    with autocast():
        # Forward pass
        coords, heatmaps = model(input_var)

        # Per-location euclidean losses
        euc_losses = dsntnn.euclidean_losses(coords, target_var)
        # Per-location regularization losses
        reg_losses = dsntnn.js_reg_losses(heatmaps, target_var, sigma_t=1.0)
        # Combine losses into an overall loss
        loss = dsntnn.average_loss(euc_losses + reg_losses)

    # loss.backward()
    # amp backward pass instead:
    scaler.scale(loss).backward()

    # Update model parameters with RMSprop
    # optimizer.step()
    # amp unscales gradients, skips stepping if NaNs
    scaler.step(optimizer)

    # update scale for next iteration
    scaler.update()

# Predictions after training
print('Predicted coords: {:0.4f}, {:0.4f}'.format(*list(coords[0, 0])))
plt.imshow(heatmaps[0, 0].detach().cpu().numpy())
plt.show()
