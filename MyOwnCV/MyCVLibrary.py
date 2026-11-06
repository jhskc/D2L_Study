import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision 
import torch.nn as nn
from torchvision.transforms import v2
from matplotlib.patches import Rectangle


def show_image(image):
    """
    Args:
        image: Is of dimensions (C, H, W)

    Notes: 
        - Assumes input image was normalized by: y = (x-0.5)/(0.5)
        - matplotlib.pyplot.imshow() expects inputs of shape (H, W, C)
        and 0-1 for float and 0-255 for int
    """
    image = image / 2 + 0.5     # unnormalize
    numpy_image = image.numpy()
    plt.imshow(np.transpose(numpy_image, (1, 2, 0)))
    plt.show()


class AllCNNParent(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential()

    def forward(self, input):
        return self.net(input)
    
    def apply_param_init(self, dummy_input, init_scheme=None):
        self.forward(*dummy_input)
        if init_scheme is not None:
            self.net.apply(init_scheme)
    
    def print_layer_summary(self, input_shape):
        X = torch.randn(size=(1, *input_shape))
        with torch.inference_mode():
            for layer in self.net:
                X = layer(X)
                print(layer.__class__.__name__, "output shape:\t", X.shape)
    

class BasicCNN(AllCNNParent):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyConv2d(out_channels=8, kernel_size=7, stride=1, padding=3), nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.LazyConv2d(out_channels=32, kernel_size=5, stride=1, padding=2), nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.LazyConv2d(out_channels=128, kernel_size=5, stride=1, padding=2), nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(), 
            nn.LazyLinear(out_features=120), nn.ReLU(), 
            nn.LazyLinear(84), nn.ReLU(), 
            nn.LazyLinear(10)    
        )


def init_cnn_xavier_uniform(module):
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)


def init_cnn_xavier_normal(module):
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_normal_(module.weight)


def init_cnn_kaiming_uniform(module):
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.kaiming_uniform_(module.weight)


def init_cnn_kaiming_normal(module):
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.kaiming_normal_(module.weight)


class CIFAR10DataSet:
    def __init__(self, batch_size=128, num_workers=4, resize: bool=False, new_size=(224, 224), 
                 persistent_workers=True, pin_memory=True, train_augments=None):
        """
        Note: Transform augments should be before normalization"""
        train_transforms_list = [v2.ToImage()] # (C, H, W), range=[0, 255] of torch.uint8]
        val_transforms_list   = [v2.ToImage()] # (C, H, W), range=[0, 255] of torch.uint8]

        if train_augments is not None:
            train_transforms_list.extend(train_augments)
        if resize is True:
            train_transforms_list.append(v2.Resize(new_size))
            val_transforms_list.append(v2.Resize(new_size))
        
        train_transforms_list.extend([v2.ToDtype(torch.float32, scale=True), # range=[0, 1] 
                                      v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        val_transforms_list.extend([v2.ToDtype(torch.float32, scale=True), # range=[0, 1] 
                                    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_transforms, val_transforms = v2.Compose(train_transforms_list), v2.Compose(val_transforms_list)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.training_set = torchvision.datasets.CIFAR10(
            root='../data', train=True, download=True, transform=train_transforms)
        self.val_set = torchvision.datasets.CIFAR10(
            root='../data', train=False, download=True, transform=val_transforms)

    def dataloader(self, train: bool):
        dataset_split = self.training_set if train else self.val_set
        dataloader = torch.utils.data.DataLoader(
            dataset_split, batch_size=self.batch_size, shuffle=train, 
            num_workers=self.num_workers, persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory)
        return dataloader
        
    def input_shape(self):
        cifar10_image, _ = next(iter(self.dataloader(train=True)))
        return cifar10_image.shape
    
    def get_classes(self):
        return self.classes


class CNNTrainer:
    def __init__(self, model, dataset, max_epochs: int=10, lr=1e-2, momentum=0.85, weight_decay=1e-5):
        """
        Note, assumes single GPU setup. 
        Built for personal use...
        """
        self.model = model
        self.dataset = dataset
        self.max_epochs = max_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device) # Need to ensure model is on GPU asw (assuming if available) 
        self.training_dataloader = self.dataset.dataloader(train=True)
        self.val_dataloader = self.dataset.dataloader(train=False)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    def train(self):
        for epoch in range(self.max_epochs):
            running_train_loss = 0.0
            running_val_loss   = 0.0
            val_acc_correct    = 0
            val_acc_total      = 0

            self.model.train()
            for (images, labels) in self.training_dataloader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()
                predictions = self.model(images)
                train_loss = self.loss_function(predictions, labels)
                train_loss.backward()
                self.optimizer.step()
                running_train_loss += train_loss.item()

            self.model.eval()
            with torch.inference_mode():
                for (images, labels) in self.val_dataloader:
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                    predictions = self.model(images)
                    val_loss = self.loss_function(predictions, labels)
                    running_val_loss += val_loss

                    _, predicted_class = torch.max(predictions, 1)
                    val_acc_total += labels.size(0)
                    val_acc_correct += (predicted_class == labels).sum()

            avg_train_loss = running_train_loss / len(self.training_dataloader)
            avg_val_loss = running_val_loss / len(self.val_dataloader)
            val_acc_pc = 100* val_acc_correct / val_acc_total
            print(f"Epoch: {epoch+1} Training loss: {avg_train_loss:.2f} Val Loss: {avg_val_loss:.2f} Val Accuracy: {(val_acc_pc):.2f}%")


class MyAlexNet(AllCNNParent):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyConv2d(out_channels=96, kernel_size=11, stride=4, padding=1), nn.ReLU(), 
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LazyConv2d(out_channels=256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LazyConv2d(out_channels=384, kernel_size=3, padding=1), nn.ReLU(),
            nn.LazyConv2d(out_channels=384, kernel_size=3, padding=1), nn.ReLU(),
            nn.LazyConv2d(out_channels=256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), 
            
            nn.Flatten(),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(p=0.4),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(p=0.4),
            nn.LazyLinear(10)
            )
        

def vgg_block(out_channels, num_convs=2, kernel_size=3, padding=1):
    """
    Notice - downsamples spatial dimensions by half every VGG block
    """
    vgg_block_layers = []
    for _ in range(num_convs):
        vgg_block_layers.append(nn.LazyConv2d(out_channels=out_channels, kernel_size=kernel_size, 
                                    stride=1, padding=padding))
        vgg_block_layers.append(nn.ReLU())
    vgg_block_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*vgg_block_layers)


class MyVGGNet(AllCNNParent):
    def __init__(self, model_arch:list[tuple]):
        super().__init__()
        conv_layers = []
        for (out_channels, num_convs, kernel_size) in model_arch:
            conv_layers.append(vgg_block(out_channels=out_channels, num_convs=num_convs, kernel_size=kernel_size))

        self.net = nn.Sequential(
            *conv_layers, 
            nn.Flatten(), 
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(p=0.5), 
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(p=0.5), 
            nn.LazyLinear(10)
        )


class MyInceptionBlock(nn.Module):
    """
    Notes:
        - Need to specify stride=1 in MaxPool2d of branch4. Need to ensure spatial
        dimensions are preserved so can concatenate along channel dimension. Otherwise, 
        assumes stride=kernel_size. Note, LazyConv2d defaults to stride=1. 
        - Forgot to include ReLU's after the 1st and before 2nd ConvLayers in branch 2 & 3. 

    """
    def __init__(self, out_chn_1: int, out_chn_2: tuple, 
                 out_chn_3: tuple, out_chn_4: int):
        super().__init__()
        self.branch1 = nn.Sequential(nn.LazyConv2d(out_channels=out_chn_1, kernel_size=1), 
                                    nn.ReLU())
        self.branch2 = nn.Sequential(nn.LazyConv2d(out_channels=out_chn_2[0], kernel_size=1),
                                     nn.ReLU(), 
                                     nn.LazyConv2d(out_channels=out_chn_2[1], kernel_size=3, padding=1),
                                     nn.ReLU())
        self.branch3 = nn.Sequential(nn.LazyConv2d(out_channels=out_chn_3[0], kernel_size=1), 
                                     nn.ReLU(),
                                     nn.LazyConv2d(out_channels=out_chn_3[1], kernel_size=5, padding=2),
                                     nn.ReLU())
        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1), 
                                     nn.LazyConv2d(out_channels=out_chn_4, kernel_size=1), 
                                     nn.ReLU())

    def forward(self, input):
        """
        Need to overwrite forward method to concatenate different branches.
        Also does not have everything in a single self.net = nn.Sequential() block.
        Concatenates alond dim=1, corresponding to channel dimension. 
        """
        return torch.cat((self.branch1(input), self.branch2(input), 
                          self.branch3(input), self.branch4(input)), dim=1)
    

class MyGoogLeNet(AllCNNParent):
    """Model architecture/layout taken from Table 1 of GoogLeNet paper.
    Assumes input of spatial dimensions 224x224
    
    Notes:
        - Forgot to specify () for self.stem, self.inception_block_X's. Need to call
        method of class, not attribute
        - Make sure to specify kernel_size, stride, padding for all MaxPools here. Forgot 2 of em.
        - Forgot to add nn.Flatten() before final classification FC layer. Dimensions screwed up.
        - For AvgPool2d, it should be stride=1 and padding=0. Check here for formula. Want to collapse
        each channel to 1 value (GlobalAvgPooling...)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(self.stem(), 
                                 self.inception_block_1(), 
                                 self.inception_block_2(), 
                                 self.inception_block_3(), 
                                 nn.AvgPool2d(kernel_size=7, stride=1, padding=0),
                                 nn.Flatten(),
                                 nn.Dropout(p=0.4),
                                 nn.LazyLinear(out_features=10))
        
    def stem(self):
        stem_block_net = nn.Sequential(nn.LazyConv2d(out_channels=64, kernel_size=7, stride=2, padding=3), 
                                   nn.ReLU(), 
                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1), 
                                   nn.LazyConv2d(out_channels=64, kernel_size=1), nn.ReLU(),
                                   nn.LazyConv2d(out_channels=192, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        return stem_block_net
    
    def inception_block_1(self):
        incep_block_net_1 = nn.Sequential(MyInceptionBlock(64,  (96,  128), (16, 32), 32),
                                          MyInceptionBlock(128, (128, 192), (32, 96), 64), 
                                          nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        return incep_block_net_1
    
    def inception_block_2(self):
        incep_block_net_2 = nn.Sequential(MyInceptionBlock(192, (96,  208), (16, 48), 64),
                                          MyInceptionBlock(160, (112, 224), (24, 64), 64),
                                          MyInceptionBlock(128, (128, 256), (24, 64), 64),
                                          MyInceptionBlock(112, (144, 288), (32, 64), 64),
                                          MyInceptionBlock(256, (160, 320), (32, 128), 128),
                                          nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        return incep_block_net_2
    
    def inception_block_3(self):
        incep_block_net_3 = nn.Sequential(MyInceptionBlock(256, (160, 320), (32, 128), 128), 
                                          MyInceptionBlock(384, (192, 384), (48, 128), 128))
        return incep_block_net_3
    

class MyBatchNorm(nn.Module):
    """
    BN(x) = gamma hadamard frac{x - mu_b}{sigma_b} + beta
    """
    def __init__(self):
        super().__init__()


class MyLayerNorm(nn.Module):
    def __init__(self):
        super().__init__()


class MyResidualBlock(nn.Module):
    """
    Derivation Notes:
    - There is truth/ideal function F*, and model's representational capacity F
    - As model complexity is increased, can think of model's capacity F as increasing. 
    - However, as model complexity increases, no garuntee that the function F is 
    "getting closer" to F*, since it can "spread" in an undesirable/unoptimal direction
    - Ensuring each additional layer/block is garunteed or at least biased to include the
    identity mapping which helps favor the model's representational capacity to "spread outwards" 
    while centered at F_1
    - This biases the model's representational capacity to "get closer" to F*, which will 
    improves resultant model performance
    - Note, this does not limit the block's representation to just to or close to the identity,
    but makes the representation of the identity easier
    - Also has additional effect of helping gradient flow (along with parameter initialization)
    - Layer Output f(x): f(x) = x + g(x) where g(x)=residual function. Derivative of which has
    identity mapping again (along with gradient of residual function). Helps training deep networks
    
    Implementation Notes:
    - Two 3x3 ConvLayers with same number of output channels as input channels
    - Have each ConvLayer followed with BatchNorm and ReLU. After 2nd ConvLayer + BatchNorm, 
    add the residual connection then add 2nd ReLU
    - Have option to change output channel count w.r.t. input channels, so use
    residual connection should have option for 1x1 ConvLayer to ensure same channel
    dimensions as main block to ensure they can be added
    - Have option for changing stride as first ConvLayer of some residual blocks are used
    do downsample spatial dimensions with stride=2
    """
    def __init__(self, num_out_channels, change_num_out_chn=False, stride=1):
        super().__init__()
        self.change_num_out_chn = change_num_out_chn
        self.main_net = nn.Sequential(nn.LazyConv2d(out_channels=num_out_channels, kernel_size=3, 
                                                    stride=stride, padding=1), 
                                      nn.LazyBatchNorm2d(), 
                                      nn.ReLU(), 
                                      nn.LazyConv2d(out_channels=num_out_channels, kernel_size=3, 
                                                    stride=1, padding=1), 
                                      nn.LazyBatchNorm2d())
        self.residual_branch = nn.LazyConv2d(out_channels=num_out_channels, kernel_size=1, 
                                             stride=stride, padding=0)

    def forward(self, X):
        main_branch_output = self.main_net(X)
        if self.change_num_out_chn:
            residual_branch_output = self.residual_branch(X)
        else:
            residual_branch_output = X
        return torch.nn.functional.relu(main_branch_output + residual_branch_output)
    

class MyDeepResidualBlock(nn.Module):
    """
    Args:
        num_int_chns: 
        num_out_chns: 
    """
    def __init__(self, num_int_chns, num_out_chns, use_1x1_conv=False, stride=1):
        super().__init__()
        self.main_branch = nn.Sequential(nn.LazyConv2d(out_channels=num_int_chns, kernel_size=1, stride=stride, padding=0), 
                                        nn.LazyBatchNorm2d(), 
                                        nn.ReLU(), 
                                        nn.LazyConv2d(out_channels=num_int_chns, kernel_size=3, stride=1, padding=1), 
                                        nn.LazyBatchNorm2d(), 
                                        nn.ReLU(), 
                                        nn.LazyConv2d(out_channels=num_out_chns, kernel_size=1, stride=1, padding=0), 
                                        nn.LazyBatchNorm2d())
        if use_1x1_conv:
            self.residual_branch = nn.Sequential(nn.LazyConv2d(out_channels=num_out_chns, kernel_size=1, stride=stride, padding=0), 
                                                 nn.LazyBatchNorm2d())
        else:
            self.residual_branch = nn.Identity()

    def forward(self, X):
        main_branch_output = self.main_branch(X)
        residual_branch_output = self.residual_branch(X)
        return torch.nn.functional.relu(main_branch_output + residual_branch_output)
    

class MyResNeXtBlock(nn.Module):
    """
    Implementation Notes
    - No need to to make for-loop for grouped convolution, already supported by nn.LazyConv2d
    - If downsampling in spatial dimension, do so during grouped-convolution layer. Ensure 
    residual branch is downsampled spatially as well then
    - Ensure number of output channels is same in ConvLayer3 of main block and residual connection
    - (num_groups) * (num_int_channels) does not have to equal 128. Goal of that figure in ResNeXt 
    paper was find optimal cardinality given fixed time complexity (compute)
    """
    def __init__(self, num_out_chns: int, num_groups: int, 
                 num_intermediate_chns:int, change_num_out_chn=False, stride=1):
        super().__init__()
        if num_intermediate_chns % num_groups != 0:
            raise ValueError("num_intermediate_chns should be divisible by num_groups")
        self.use_projection = change_num_out_chn or stride != 1
        
        self.ConvLayer1 = nn.LazyConv2d(out_channels=num_intermediate_chns, kernel_size=1, stride=1, padding=0)
        self.BatchNorm1 = nn.LazyBatchNorm2d()
        
        self.ConvLayer2 = nn.LazyConv2d(out_channels=num_intermediate_chns, kernel_size=3, stride=stride, padding=1, groups=num_groups)
        self.BatchNorm2 = nn.LazyBatchNorm2d()

        self.ConvLayer3 = nn.LazyConv2d(out_channels=num_out_chns, kernel_size=1, stride=1, padding=0)
        self.BatchNorm3 = nn.LazyBatchNorm2d()

        if self.use_projection:
            self.ResidualConvLayer = nn.Sequential(nn.LazyConv2d(out_channels=num_out_chns, kernel_size=1, stride=stride, padding=0), 
                                                   nn.LazyBatchNorm2d())
        else:
            self.ResidualConvLayer = nn.Identity()

    def forward(self, X):
        Y = torch.nn.functional.relu(self.BatchNorm1(self.ConvLayer1(X)))
        Y = torch.nn.functional.relu(self.BatchNorm2(self.ConvLayer2(Y)))
        Y = self.BatchNorm3(self.ConvLayer3(Y))
        ResidualBranch = self.ResidualConvLayer(X)
        return torch.nn.functional.relu(Y + ResidualBranch)
        

class MyShallowResNet(AllCNNParent):
    def __init__(self, model_arch: tuple[tuple[int, int],...]):
        super().__init__()
        self.model_arch = model_arch
        self.net = nn.Sequential(self.stem(), 
                                 self.body_block(),
                                 self.head())

    def stem(self):
        stem_net = nn.Sequential(nn.LazyConv2d(out_channels=64, kernel_size=7, stride=2, padding=3), 
                                 nn.LazyBatchNorm2d(), 
                                 nn.ReLU(), 
                                 nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        return stem_net
    
    def body_block(self):
        body_modules_list = []
        for block_index, (num_res_blocks, num_out_chns) in enumerate(self.model_arch):
            for kk in range(num_res_blocks):
                if kk == 0 and block_index != 0: # First residual module within block
                    body_modules_list.append(MyResidualBlock(num_out_channels=num_out_chns, 
                                                             change_num_out_chn=True, stride=2))
                else:
                    body_modules_list.append(MyResidualBlock(num_out_channels=num_out_chns))
        return nn.Sequential(*body_modules_list)


    def head(self):
        head_net = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                 nn.Flatten(), 
                                 nn.LazyLinear(out_features=10))
        return head_net


class MyDeepResNet(MyShallowResNet):
    """Just need to rewrite the body_block() method to pass arguments for MyDeepResidualBlock"""  
    def body_block(self):
        body_modules_list = []
        for stage_index, (num_res_blocks, num_int_chns, num_out_chns) in enumerate(self.model_arch):
            for block_index in range(num_res_blocks):
                use_1x1_res = (block_index == 0)

                # Spatially downsample only for conv stages 2-4 and first residual block within each stage
                stride = (2 if block_index == 0 and stage_index != 0 else 1)
                body_modules_list.append(MyDeepResidualBlock(num_int_chns=num_int_chns, 
                                                             num_out_chns=num_out_chns, 
                                                             use_1x1_conv=use_1x1_res, 
                                                             stride=stride))
        return nn.Sequential(*body_modules_list)


class MyResNeXt(AllCNNParent):
    def __init__(self):
        super().__init__()


def show_images(images, num_rows, num_cols, figsize=None):
    """
    Display a list of images in a num_rows × num_cols grid.

    images: list of image arrays
    num_rows: number of subplot rows
    num_cols: number of subplot columns
    figsize: optional figure size, such as (12, 8)
    """
    if len(images) > num_rows * num_cols:
        raise ValueError(
            f"Grid has {num_rows * num_cols} slots, "
            f"but {len(images)} images were supplied."
        )
    if figsize is None:
        figsize = (4 * num_cols, 4 * num_rows)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    # Makes axes iterable even for a 1×1 grid
    axes = np.atleast_1d(axes).ravel()

    for ax, image in zip(axes, images):
        ax.imshow(image)
        ax.axis("off")

    # Hide unused subplots
    for ax in axes[len(images):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def apply_transforms(image, augments, num_rows=3, num_cols=4):
    """
    Args:
        image: Singular image to perform augments on 
        augments: torchvision.transforms.v2.Compose object
    """
    if isinstance(image, np.ndarray):
        image = torchvision.transforms.ToPILImage()(image)
    augmented_images = []
    for _ in range(num_rows * num_cols):
        augmented_images.append(augments(image))
    show_images(augmented_images, num_rows=num_rows, num_cols=num_cols)


class FineTuneTrainer(CNNTrainer):
    def __init__(self, model, dataset, max_epochs: int=5, lr=1e-2, momentum=0.9, weight_decay=1e-5, 
                 freeze_pretrained: bool=True, pretrained_lr_multiplier: float=0.1):
        """
        Fine tune a pretrained model whose last linear layer has already been swapped out.

        Args:
            model: Pretrained model, like a torchvision.models.resnet18(
                weights=torchvision.models.ResNet18_Weights.DEFAULT). NOTE, the last layer should
                already be swapped out and initiailzied to nn.Linear instance and named "fc"
            dataset: Should be an object that has the same interface as the CIFAR10DataSet class above 
                (say, .dataloader() method)
            freeze_pretrained: Boolean to set if pretrained model weights are frozen or not during finetuing
            pretrained_lr_multiplier: Pretrained weights update with: lr * pretrained_lr_multiplier, 
                while model.fc (new, last linear layer) updates at lr.
        Implementation Notes:
            - No need to rewrite entire training loop, just use the one from CNNTrainer. Just swap out
            self.optimizer with corresponding adjusted learning rates (in case of freeze_pretrained=False)
            - Need to pass arguments through super().__init__() to get them thru to __init__ method 
            of CNNTrainer
            - Use filter() function to only keep tesnors where: p.requires_grad is True and pass em to optimizer
        """
        assert hasattr(model, "fc") and isinstance(model.fc, nn.Linear), (
            "model.fc must exist and be named as such. Should also be nn.Linear instance.")

        if freeze_pretrained:
            model.requires_grad_(False)
            model.fc.requires_grad_(True)

        super().__init__(model=model, dataset=dataset, max_epochs=max_epochs, 
                         lr=lr, momentum=momentum, weight_decay=weight_decay)

        if freeze_pretrained:
            self.optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()), 
                lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            fc_param_ids = {id(p) for p in self.model.fc.parameters()}
            pretrained_params = [p for p in self.model.parameters() if id(p) not in fc_param_ids]
            self.optimizer = torch.optim.SGD(
                [{"params": self.model.fc.parameters(), "lr": lr},
                 {"params": pretrained_params, "lr": lr * pretrained_lr_multiplier}], 
                momentum=momentum, weight_decay=weight_decay)
            

def convert_corners_to_center(box_corners):
    """
    Note: 
        y-axis is top-to-bottom, while x-axis is left-to-right
    Args:
        box_corners: Tensor of shape (N, 4) where each row corresponds to a box, where each 
            entry of the row corresponds to (x1, y1, x2, y2) where (x1, y1) is the upper-left
            corner, and (x2, y2) corresponds to the lower-right corner. 
    Returns:
        box_centers: Tensor, also of shape (N, 4) where there are N rows, where each row is 
            (c_x, c_y, w, h) corresponding to the coordinates of the center, and the total width
            and height of the total box (not half height/width)
    """
    x1 = box_corners[:, 0]
    y1 = box_corners[:, 1]
    x2 = box_corners[:, 2]
    y2 = box_corners[:, 3]
    c_x = (x1 + x2) / 2
    c_y = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    box_centers = torch.stack((c_x, c_y, w, h), dim=1)
    return box_centers


def convert_center_to_corners(box_centers):
    """
    Note: 
        y-axis is top-to-bottom, while x-axis is left-to-right
    Args:
        box_centers: Tensor of shape (N, 4) where there are N rows, where each row is 
            (c_x, c_y, w, h) corresponding to the coordinates of the center, and the total width
            and height of the total box (not half height/width)
    Returns:
        box_corners: Tensor of shape (N, 4) where each row corresponds to a box, where each 
            entry of the row corresponds to (x1, y1, x2, y2) where (x1, y1) is the upper-left
            corner, and (x2, y2) corresponds to the lower-right corner. 
    """
    c_x = box_centers[:, 0]
    c_y = box_centers[:, 1]
    w   = box_centers[:, 2]
    h   = box_centers[:, 3]
    x1 = c_x - 0.5 * w
    y1 = c_y - 0.5 * h
    x2 = c_x + 0.5 * w
    y2 = c_y + 0.5 * h
    box_corners = torch.stack((x1, y1, x2, y2), dim=1)
    return box_corners


def bounding_box_center_to_plt(box_corners, color):
    """
    Note: Can only be used for a single bounding box, can't feed a tensor of multiple bounding
        boxes into it. 
    Args:
        box_centers: Iterable (not necessarily tensor) of shape (1, 4) where it corresponds to
            (upper_left_x, upper_left_y, lower_right_x, lower_right_y) to format usable by 
            Rectangle object of: ((upper_left_x, upper_left_y), width, height)
    """
    width  = box_corners[2] - box_corners[0]
    height = box_corners[3] - box_corners[1]
    return Rectangle((box_corners[0], box_corners[1]), width=width, height=height, 
                     fill=False, color=color, linewidth=1)


def multibox_prior(data, sizes, ratios):
    """
    Args: 

    """


def show_bounding_boxes(axes, bounding_boxes, labels=None, colors=None):
    return NotImplementedError


def box_intersection_over_union(boxes1, boxes_2):
    """
    Args:
        boxes1: 
        boxes2: 
    """

def assign_anchor_to_bounding_box(ground_truth, anchors, device, iout_threshold=0.5):
    """
    """
    return NotImplementedError


def offset_boxes(anchors, assigned_bounding_box, eps=1e-6):
    """
    """
    return NotImplementedError


def multibox_target(anchors, labels):
    """
    """
    return NotImplementedError


def offset_inverse(anchors, offset_predictions):
    """
    """
    return NotImplementedError


def NonMaxSuppression(boxes, scores, iou_threshold):
    """
    """
    return NotImplementedError


def multibox_detection(class_probs, offset_predictions, anchor, nms_threshold=0.5, pos_threshold=0.009999999):
    """
    """
    raise NotImplementedError
    

class VisionTransformerMLP(nn.Module):
    def __init__(self, num_hiddens, num_outputs, dropout=0.5):
        """
        Implementation Notes:
        - MLP block has 2 layers with a single GELU non-linearity. 
        - LayerNorm normalization applied before the MLP block (and
        also before the Multi-Head Self-Attention block, thus not 
        included here)
        """
        super().__init__()
        self.net = nn.Sequential(nn.LazyLinear(out_features=num_hiddens), 
                                 nn.GELU(), nn.Dropout(p=dropout), 
                                 nn.LazyLinear(out_features=num_outputs),
                                 nn.Dropout(p=dropout))
        
    def forward(self, X):
        return self.net(X)


class ImagePatchEmbed(nn.Module):
    def __init__(self, image_size=64, patch_size=16, dim_embed=512):
        """
        Implementation Notes:
        - Convolution layer with number of output channels as dim_embed where both
        the kernel_size and stride are set to patch_size to create non-overlapping
        "snapshots" with the kernel 
        """
        super().__init__()
        def return_tuple(x):
            if not isinstance(x, (list, tuple)):
                return (x, x)
            return x

        image_size = return_tuple(image_size)
        patch_size = return_tuple(patch_size)

        assert image_size[0] % patch_size[0] == 0
        assert image_size[1] % patch_size[1] == 0

        self.grid_height = image_size[0] // patch_size[0]
        self.grid_width  = image_size[1] // patch_size[1]

        self.num_patches = ((image_size[0] // patch_size[0]) * 
                            (image_size[1] // patch_size[1]))
        self.ConvProjection = nn.Conv2d(in_channels=3, out_channels=dim_embed, 
                                        kernel_size=patch_size, stride=patch_size)
        
    def forward(self, X):
        """
        Implementation Notes:
        - ConvLayer will return: (B, C, H, W). Want to flatten spatial dims.
        - Then, swap the channels and flattened spatial dims. We want projected
        embedding vector last, and H_out, W_out of the ConvLayer will correspond
        to number of vertical and horizontal patches, flattening will combine all 
        patches to a single dimension. 
        """
        projection_output = self.ConvProjection(X) # Line Output: (B, C_out, H_out, W_out)
        combined_patches = projection_output.flatten(start_dim=2, end_dim=3) # Line Output: (B, C_out, num_patches)
        transposed = combined_patches.transpose(1, 2) # Line Output: (B, num_patches, C_out)
        return transposed
    

class VisionTransformerBlock(nn.Module):
    def __init__(self, dim_embed, mlp_num_hidden, 
                 num_attn_heads, dropout_prob, use_bias=False):
        super().__init__()
        assert dim_embed % num_attn_heads == 0

        self.LayerNorm1 = nn.LayerNorm(normalized_shape=dim_embed)

        self.MHSAttention = nn.MultiheadAttention(embed_dim=dim_embed, num_heads=num_attn_heads, 
                                                   dropout=dropout_prob, bias=use_bias, batch_first=True)
        # self.MyMHAttention = 

        self.LayerNorm2 = nn.LayerNorm(normalized_shape=dim_embed)
        self.MLP = VisionTransformerMLP(num_hiddens=mlp_num_hidden, num_outputs=dim_embed, dropout=dropout_prob)

    def forward(self, X):
        """
        Implementation Notes
        - Discard _ output of MHSA. Don't need weights themselves, onlt the transformation
        - Reference Fig. 1 of ViT 16x16 Image
        - Pre-LayerNorm (1) before Multi-Head Self-Attention (MHSA)
        - Residual connection from before LayerNorm (1) to output of MHSA
        - Pre-LayerNorm (2) before MLP Block
        - Residual connection from before LayerNorm (2) to output of MLP
        - Output of current transformer encoder block
        """
        Y = self.LayerNorm1(X)
        attn_output_transformation, _ = self.MHSAttention(query=Y, key=Y, value=Y, need_weights=False)
        X = X + attn_output_transformation

        A = self.LayerNorm2(X)
        B = self.MLP(A)
        block_output = X + B
        return block_output
    

class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, dim_embed, mlp_num_hiddens, num_attn_heads, num_ViT_blocks, 
                emb_dropout_prob, mlp_dropout_prob, lr=0.01, use_bias=False, num_classes=10):
        """
        Implementation Notes:
        - Use learned positional embeddings for now, since paper says they didn't notice any
        difference between fixed/learned. 
        - NOTE - come back later and add versions/options for 
        trigonometric embedding and rotational positional embedding
        """
        super().__init__()
        self.patch_embeddings = ImagePatchEmbed(image_size=image_size, patch_size=patch_size, 
                                                dim_embed=dim_embed)
        self.class_token = nn.Parameter(torch.zeros(1, 1, dim_embed))
        num_tokens = self.patch_embeddings.num_patches + 1 
        self.position_embedding = nn.Parameter(torch.randn(1, num_tokens, dim_embed))
        self.dropout = nn.Dropout(p=emb_dropout_prob)
        
        self.StackedTransformerEncoders = nn.Sequential()
        for jj in range(num_ViT_blocks):
            self.StackedTransformerEncoders.add_module(f"{jj}", VisionTransformerBlock(
                dim_embed=dim_embed, mlp_num_hidden=mlp_num_hiddens, num_attn_heads=num_attn_heads,
                dropout_prob=mlp_dropout_prob, use_bias=use_bias
            ))
        self.classification_head = nn.Sequential(nn.LayerNorm(normalized_shape=dim_embed), 
                                                 nn.Linear(in_features=dim_embed, out_features=num_classes))

    def forward(self, X):
        # Pass raw image through patch embeddings
        X = self.patch_embeddings(X)

        # Concatenate class token at first position
        # Expand the class_token from: (1, 1, dim_embed) to: (batch_size, 1, dim_embed). The -1 mean to keep unchanged
        # Concatenate by putting class token in front of patch embedding tokens and concatenating along token dimension
        # Note: .expand() is used to create a new view of it by treating same parameter(s) as appearing 
        # multiple times across batches without copying its data. Backprop will accumulate all gradients across
        # batch samples back into single class token vector
        X = torch.cat((self.class_token.expand(X.shape[0], -1, -1), X), dim=1)

        # Add positional embedding to embedding vectors
        X = self.dropout(X + self.position_embedding)

        # Pass embeddings through all transformer encoder blocks
        for block in self.StackedTransformerEncoders:
            X = block(X)

        output = self.classification_head(X[:, 0, :]) # Line Input: (batch_size, dim_embed). Line Output: (batch_size, num_classes)
        return output


class MyScaledDotProductAttention(nn.Module):
    # NOTE: This needs to be above/earlier the MuMultiHeadAttention class, otherwise will register as missing/undefined
    def __init__(self):
        super().__init__()

    def forward(self, queries, keys, values):
        """ 
        Note, for purposes of square image in ViT: num_queries = num_keys = num_values
        Args:
            queries: Of shape (batch_size * num_heads, num_queries, dim_embed / num_heads)
            keys:    Of shape (batch_size * num_heads, num_keys, dim_embed / num_heads)
            values:  Of shape (batch_size * num_heads, num_values, dim_embed / num_heads)
        """
        scale_factor = queries.shape[-1] ** -0.5 # Extract 1/sqrt(dim_head)
        attn_scores = torch.bmm(input=queries, mat2=keys.transpose(1, 2)) * scale_factor # Output Shape: (batch_size * num_heads, num_queries, num_keys)
        normed_attn_scores = torch.nn.functional.softmax(input=attn_scores, dim=2) # Want to normalized along key dimension to find how much each key corresponds to each query. Equivalent to dim=-1\
        weighted_values = torch.bmm(input=normed_attn_scores, mat2=values) # Line Input Shape: (batch_size * num_heads, num_queries, num_keys) x (batch_size * num_heads, num_values, dim_embed / num_head). Since this is self-attention, num_keys = num_values. 
        return weighted_values
    

class MyMultiHeadAttention(nn.Module):
    """
    Input: (batch_size, num_patches, C_out) corresponding to 
           (batch_size, num_patches, flatteened_image_patch) where num_patches (along
           with an additional class token will be used as the query, key, and value vectors)
    """
    def __init__(self, dim_embed, num_heads, use_embed_bias=False):
        super().__init__()
        assert dim_embed % num_heads == 0
        self.num_heads = num_heads
        self.dim_embed = dim_embed
        self.dim_head  = dim_embed // num_heads
        self.DotProductAttention = MyScaledDotProductAttention()
        self.W_query  = nn.LazyLinear(out_features=dim_embed, bias=use_embed_bias)
        self.W_key    = nn.LazyLinear(out_features=dim_embed, bias=use_embed_bias)
        self.W_value  = nn.LazyLinear(out_features=dim_embed, bias=use_embed_bias)
        self.W_output = nn.LazyLinear(out_features=dim_embed, bias=use_embed_bias)

    def forward(self, query_vectors, key_vectors, value_vectors):
        """
        1. Project down to embedding dimensions using weights defined in __init_-
        2. Split embedding vector equally among heads (use assert to ensure clean division)
        3. Make sure to reshape along batch_size * num_heads for attention operation parallelization
        4. Run through scale dot product attention using an instance of class from __init__
        5. Concatenate output by reshaped back into shape: (batch_size, num_queries, dim_embed)
        6. Final linear projection layer w/ W_output
        """
        proj_queries = self.W_query(query_vectors)
        proj_keys    = self.W_key(key_vectors)
        proj_values  = self.W_value(value_vectors)

        #def _split_vec_across_heads(X):
        #    """Reshape Input: (batch_size, num_query, dim_embed) 
        #           to Output: (batch_size * num_head, num_query, dim_embed/num_heads)"""
        #    batch_size = X.shape[0]
        #    num_query  = X.shape[1]
        #    dim_embed  = X.shape[2]
        #    split_vec = X.reshape(batch_size * self.num_heads, num_query, int(dim_embed / self.num_heads))
        #    return split_vec
        def _split_vec_across_heads(X):
            """Original implemenation (the one above) messes up the order when
            immediately reshaping to the final desired shape"""
            batch_size, num_tokens, dim_embed = X.shape

            X = X.reshape(batch_size, num_tokens, self.num_heads, self.dim_head) 
            Y = X.transpose(1, 2) # Line Output: (batch_size, num_heads, num_queries, dim_embed/num_heads)
            Z = Y.reshape(batch_size * self.num_heads, num_tokens, self.dim_head) # Line Output: (batch_size * num_heads, num_queries, dim_embed / num_head)
            return Z 

        attn_queries = _split_vec_across_heads(proj_queries)
        attn_keys    = _split_vec_across_heads(proj_keys)
        attn_values  = _split_vec_across_heads(proj_values)
        attn_output  = self.DotProductAttention(queries=attn_queries, keys=attn_keys, values=attn_values)

        def _concatenate_heads(X):
            """Input Shape: (batch_size * num_heads, num_queries, dim_embed / num_heads)"""
            X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
            X = X.transpose(1, 2) # Line Output Shape: (batch_size, num_queries, num_heads, dim_embed / num_heads)
            X = X.reshape(X.shape[0], X.shape[1], -1) # Line Output Shape: (batch_size, num_queries, dim_embed)
            return X
        attn_output_concate = _concatenate_heads(attn_output)
        output = self.W_output(attn_output_concate)
        return output


class TestVisionTransformerBlock(nn.Module):
    """
    Same thing as VisionTransformerBlock above, but with the nn.MultiheadAttention swapped out for my own 
    implementation to test if it works. 
    """
    def __init__(self, dim_embed, mlp_num_hidden, 
                 num_attn_heads, dropout_prob, use_bias=False):
        super().__init__()
        assert dim_embed % num_attn_heads == 0

        self.LayerNorm1 = nn.LayerNorm(normalized_shape=dim_embed)

        # self.MHSAttention = nn.MultiheadAttention(embed_dim=dim_embed, 
        #                                          num_heads=num_attn_heads, 
        #                                          dropout=dropout_prob, 
        #                                          bias=use_bias, 
        #                                          batch_first=True)
        self.MyMultiHeadAttention = MyMultiHeadAttention(dim_embed=dim_embed, num_heads=num_attn_heads, use_embed_bias=use_bias)

        self.LayerNorm2 = nn.LayerNorm(normalized_shape=dim_embed)
        self.MLP = VisionTransformerMLP(num_hiddens=mlp_num_hidden, num_outputs=dim_embed, dropout=dropout_prob)

    def forward(self, X):
        Y = self.LayerNorm1(X)
        attn_output_transformation = self.MyMultiHeadAttention(query_vectors=Y, key_vectors=Y, value_vectors=Y) # , need_weights=False)
        X = X + attn_output_transformation

        A = self.LayerNorm2(X)
        B = self.MLP(A)
        block_output = X + B
        return block_output
    

class TestVisionTransformer(nn.Module):
    """
    Same thing as VisionTransformer above, but calls the TestVisionTransformerBlock which swaps out 
    nn.MultiheadAttention swapped out for my own implementation to test if it works. 
    """
    def __init__(self, image_size, patch_size, dim_embed, mlp_num_hiddens, num_attn_heads, num_ViT_blocks, 
                emb_dropout_prob, mlp_dropout_prob, lr=0.01, use_bias=False, num_classes=10):
        super().__init__()
        self.patch_embeddings = ImagePatchEmbed(image_size=image_size, patch_size=patch_size, 
                                                dim_embed=dim_embed)
        self.class_token = nn.Parameter(torch.zeros(1, 1, dim_embed))
        num_tokens = self.patch_embeddings.num_patches + 1 
        self.position_embedding = nn.Parameter(torch.randn(1, num_tokens, dim_embed))
        self.dropout = nn.Dropout(p=emb_dropout_prob)
        
        self.StackedTransformerEncoders = nn.Sequential()
        for jj in range(num_ViT_blocks):
            self.StackedTransformerEncoders.add_module(f"{jj}", TestVisionTransformerBlock(
                dim_embed=dim_embed, mlp_num_hidden=mlp_num_hiddens, num_attn_heads=num_attn_heads,
                dropout_prob=mlp_dropout_prob, use_bias=use_bias))
        self.classification_head = nn.Sequential(nn.LayerNorm(normalized_shape=dim_embed), 
                                                 nn.Linear(in_features=dim_embed, out_features=num_classes))

    def forward(self, X):
        X = self.patch_embeddings(X)
        X = torch.cat((self.class_token.expand(X.shape[0], -1, -1), X), dim=1) 
        X = self.dropout(X + self.position_embedding)
        for block in self.StackedTransformerEncoders:
            X = block(X)
        output = self.classification_head(X[:, 0, :])
        return output


class Axial2DRotaryPositionEmbedding(nn.Module):
    """Apply 2D axial RoPE to the query and key vectors of each attention head.

    The per-head embedding dimension is split into two equal subspaces:
        first half  -> horizontal/column position
        second half -> vertical/row position

    Each half is then divided into adjacent pairs and rotated by position-
    dependent angles. Classification token left unchanged since have 
    no spatial/physical location.

    Args:
        dim_head: Number of features in one attention head. Must be divisible by 4
        grid_size: Tuple (grid_height, grid_width) 
        base: Base used to construct the geometric sequence of RoPE frequencies. 
            Default set to 10,000 as in papers. 
    """
    def __init__(self, dim_head, grid_size, base=10000.0):
        super().__init__()

        assert dim_head % 4 == 0

        if not isinstance(grid_size, (tuple, list)) or len(grid_size) != 2:
            raise ValueError("grid_size must be a tuple or list of len 2 of format (grid_height, grid_width).")

        grid_height, grid_width = int(grid_size[0]), int(grid_size[1])

        self.dim_head    = dim_head
        self.axis_dim    = dim_head // 2 
        self.grid_height = grid_height
        self.grid_width  = grid_width
        self.num_spatial_tokens = grid_height * grid_width

        # One frequency is used for every adjacent feature pair in one axis of each head. Shape: (axis_dim / 2,)
        # Frequencies = 10000^-(index/axis_dim) ; where index = 0, 2, 4, ..., axis_dim-2
        inverse_frequencies = 1.0 / (base ** (torch.arange(start=0, end=self.axis_dim, step=2, dtype=torch.float32) / self.axis_dim))

        # Have both row and column positions return a single 1D tensor w/ .repeat_interleave() & .repeat()
        # Recall: .repeat() repeats entire sequence specified number of times, while .repeat_interleave()
        # will repeat each element specified number of times (i.e. stretched)
        row_positions = torch.arange(start=0, end=grid_height, step=1, dtype=torch.float32).repeat_interleave(repeats=grid_width)
        col_positions = torch.arange(start=0, end=grid_width,  step=1, dtype=torch.float32).repeat(grid_height)

        # Compute outer products for computational efficiency instead of loops
        # Shape: (num_spatial_tokens, axis_dim / 2)
        row_angles = torch.outer(row_positions, inverse_frequencies)
        col_angles = torch.outer(col_positions, inverse_frequencies)

        # Use register_buffer() here since they are are non-trainable (trig rotation values 
        # shoud be fixed), move with model.to(device), and are included in the state_dict 
        # so the module can be saved/loaded. Otherwise, row_cos and like are on CPU 
        # while everything is on GPU. Also, better than using .to('cuda') for all 4 tensors below
        self.register_buffer("row_cos", torch.cos(row_angles))
        self.register_buffer("row_sin", torch.sin(row_angles))
        self.register_buffer("col_cos", torch.cos(col_angles))
        self.register_buffer("col_sin", torch.sin(col_angles))
    
    # No accesses to model attributes (self.whatever), thus wrap it in this. 
    # This function does not need to access/modify the class's state, but
    # just grouped here for organizational purposes
    @staticmethod
    def _rotate_adjacent_pairs(X, cosine, sine):
        """
        Rotate adjacent feature pairs governed by 2D rotation matrices constructed
        with supplied cosine, sine tensor arguments. 

        Args:
            X: Tensor of shape (batch_size, num_heads, num_spatial_tokens, axis_dim).
            cosine, sine: Broadcastable tensors of shape (1, 1, num_spatial_tokens, axis_dim / 2).

        Returns:
            Tensor with the same shape as X.
        """
        assert X.shape[-1] % 2 == 0, "axis_dim should be divisible by 2"

        # Group adjacent features as (..., num_rotation_pairs, 2).
        # Preserve (batch_size, num_heads, num_spatial_tokens) with tupel unpacking
        X_pairs = X.reshape(*X.shape[:-1], X.shape[-1] // 2, 2)
        first_components  = X_pairs[:, :, :, :, 0]
        second_components = X_pairs[:, :, :, :, 1]

        rotated_first_components  = first_components * cosine - second_components * sine
        rotated_second_components = first_components * sine + second_components * cosine

        rotated_pairs = torch.stack((rotated_first_components, rotated_second_components), dim=-1)
        output = rotated_pairs.flatten(start_dim=-2)
        return output

    def forward(self, queries, keys):
        """Rotate query and key patch tokens. Leave <cls> token unchanged.

        Args:
            queries, keys: Tensors of shape (batch_size, num_heads, num_tokens, dim_head).

        Returns:
            Tuple ``(rotated_queries, rotated_keys)`` with the same shapes.
        """
        query_cls  = queries[:, :, : 1, :]
        key_cls    = keys[:, :, : 1, :]
        query_patches = queries[:, :, 1:, :]
        key_patches   = keys[:, :, 1:, :]

        # First half uses column position; second half uses row position.
        query_columns, query_rows = torch.split(query_patches, self.axis_dim, dim=-1)
        key_columns, key_rows     = torch.split(key_patches,   self.axis_dim, dim=-1)

        col_cos = self.col_cos[None, None, :, :]
        col_sin = self.col_sin[None, None, :, :]
        row_cos = self.row_cos[None, None, :, :]
        row_sin = self.row_sin[None, None, :, :]

        query_columns = self._rotate_adjacent_pairs(X=query_columns, cosine=col_cos, sine=col_sin)
        key_columns   = self._rotate_adjacent_pairs(X=key_columns, cosine=col_cos, sine=col_sin)
        query_rows    = self._rotate_adjacent_pairs(X=query_rows, cosine=row_cos, sine=row_sin)
        key_rows      = self._rotate_adjacent_pairs(X=key_rows, cosine=row_cos, sine=row_sin)

        rotated_query_patches = torch.cat((query_columns, query_rows), dim=-1)
        rotated_key_patches   = torch.cat((key_columns, key_rows), dim=-1)

        # Concatenate along token dimension w/ cls token first
        rotated_queries = torch.cat((query_cls, rotated_query_patches), dim=2)
        rotated_keys    = torch.cat((key_cls, rotated_key_patches), dim=2)
        return rotated_queries, rotated_keys


class MyMultiHeadAttention2DRoPE(nn.Module):
    """Custom multi-head self-attention with 2D axial RoPE.

    This class follows MyMultiHeadAttention, but applies 2D-RoPE to Q and K.
    After rotation, batch and head are merged so MyScaledDotProductAttention 
    class can be reused, unchanged.
    """

    def __init__(self, dim_embed, num_heads, grid_size, use_embed_bias=False, rope_base=10000.0):
        super().__init__()
        assert dim_embed % num_heads == 0, "dim_embed should be divisible by num_heads"

        self.num_heads = num_heads
        self.dim_embed = dim_embed
        self.dim_head = dim_embed // num_heads

        assert self.dim_head % 4 == 0, "Dimension of each head should be divisible by 4"

        self.DotProductAttention = MyScaledDotProductAttention()
        self.RotaryPositionEmbedding = Axial2DRotaryPositionEmbedding(
            dim_head=self.dim_head, grid_size=grid_size, base=rope_base)

        self.W_query  = nn.LazyLinear(out_features=dim_embed, bias=use_embed_bias)
        self.W_key    = nn.LazyLinear(out_features=dim_embed, bias=use_embed_bias)
        self.W_value  = nn.LazyLinear(out_features=dim_embed, bias=use_embed_bias)
        self.W_output = nn.LazyLinear(out_features=dim_embed, bias=use_embed_bias)

    def _split_heads(self, X):
        """Split/reshape from (B, N, D_embed) to (B, H, N, D_head)."""
        batch_size, num_tokens, _ = X.shape # _ = dim_embed
        X = X.reshape(batch_size, num_tokens, self.num_heads, self.dim_head)
        return X.transpose(1, 2)

    def _merge_batch_and_heads(self, X):
        """Reshape from (B, H, N, D_head) to (B * H, N, D_head)."""
        batch_size, num_heads, num_tokens, dim_head = X.shape
        return X.reshape(batch_size * num_heads, num_tokens, dim_head)

    def _concatenate_heads(self, X):
        """Reshape from (B * H, N, D_head) back to (B, N, D_embed)."""
        batch_heads, num_tokens, _ = X.shape # _ = dim_head
        batch_size = batch_heads // self.num_heads
        X = X.reshape(batch_size, self.num_heads, num_tokens, self.dim_head)
        X = X.transpose(1, 2)
        return X.reshape(batch_size, num_tokens, self.dim_embed)

    def forward(self, query_vectors, key_vectors, value_vectors):
        """Run multi-head attention with RoPE applied only to Q and K."""
        projected_queries = self.W_query(query_vectors)
        projected_keys    = self.W_key(key_vectors)
        projected_values  = self.W_value(value_vectors)

        attention_queries = self._split_heads(projected_queries)
        attention_keys    = self._split_heads(projected_keys)
        attention_values  = self._split_heads(projected_values)

        attention_queries, attention_keys = self.RotaryPositionEmbedding(
            queries=attention_queries, keys=attention_keys)

        attention_queries = self._merge_batch_and_heads(attention_queries)
        attention_keys    = self._merge_batch_and_heads(attention_keys)
        attention_values  = self._merge_batch_and_heads(attention_values)

        attention_output = self.DotProductAttention(queries=attention_queries,
            keys=attention_keys, values=attention_values)

        concatenated_output = self._concatenate_heads(attention_output)
        output_layer = self.W_output(concatenated_output)
        return output_layer


class VisionTransformerBlock2DRoPE(nn.Module):
    """ViT encoder block using MyMultiHeadAttention2DRoPE."""
    def __init__(self, dim_embed, mlp_num_hidden, num_attn_heads, dropout_prob,
        grid_size, use_bias=False, rope_base=10000.0):
        super().__init__()

        assert dim_embed % num_attn_heads == 0, "dim_embed should be divisible by num_attn_heads"

        self.LayerNorm1 = nn.LayerNorm(normalized_shape=dim_embed)
        self.MyMultiHeadAttention2DRoPE = MyMultiHeadAttention2DRoPE(
            dim_embed=dim_embed, num_heads=num_attn_heads, grid_size=grid_size,
            use_embed_bias=use_bias, rope_base=rope_base)

        self.LayerNorm2 = nn.LayerNorm(normalized_shape=dim_embed)
        self.MLP = VisionTransformerMLP(num_hiddens=mlp_num_hidden, num_outputs=dim_embed,
            dropout=dropout_prob)

    def forward(self, X):
        normalized_attention_input = self.LayerNorm1(X)
        attention_output = self.MyMultiHeadAttention2DRoPE(
            query_vectors=normalized_attention_input,
            key_vectors=normalized_attention_input,
            value_vectors=normalized_attention_input)
        
        X = X + attention_output

        normalized_mlp_input = self.LayerNorm2(X)
        mlp_output = self.MLP(normalized_mlp_input)
        return X + mlp_output


class VisionTransformer2DRoPE(nn.Module):
    """Vision Transformer using 2D axial RoPE instead of additive positions.

    The class reuses ImagePatchEmbed and VisionTransformerMLP, but swaps out previous
    MyMultiHeadAttention for MyMultiHeadAttention2DRoPE for positional embeddings. 
    Thus, no learnable position_embedding nn.Parameter()
    separate RoPE-aware attention and encoder-block classes. There is no learned
    ``position_embedding`` parameter: spatial position is injected into every
    block through the rotated queries and keys.
    """
    def __init__(self, image_size, patch_size, dim_embed, mlp_num_hiddens, num_attn_heads,
        num_ViT_blocks, emb_dropout_prob, mlp_dropout_prob, use_bias=False, num_classes=10,
        rope_base=10000.0):
        super().__init__()

        def _return_tuple(value):
            if isinstance(value, (tuple, list)):
                if len(value) != 2:
                    raise ValueError("image_size and patch_size tuples must have length 2.")
                return (int(value[0]), int(value[1]))
            return (int(value), int(value))

        image_height, image_width = _return_tuple(image_size)
        patch_height, patch_width = _return_tuple(patch_size)

        self.image_size = (image_height, image_width)
        self.patch_size = (patch_height, patch_width)
        self.grid_size  = (image_height // patch_height, image_width // patch_width)

        self.patch_embeddings = ImagePatchEmbed(image_size=self.image_size,
            patch_size=self.patch_size, dim_embed=dim_embed)
        self.class_token = nn.Parameter(torch.zeros(1, 1, dim_embed))
        self.dropout = nn.Dropout(p=emb_dropout_prob)

        self.StackedTransformerEncoders = nn.Sequential()
        for block_index in range(num_ViT_blocks):
            self.StackedTransformerEncoders.add_module(
                f"{block_index}",
                VisionTransformerBlock2DRoPE(dim_embed=dim_embed, mlp_num_hidden=mlp_num_hiddens,
                    num_attn_heads=num_attn_heads, dropout_prob=mlp_dropout_prob, grid_size=self.grid_size,
                    use_bias=use_bias, rope_base=rope_base))

        self.classification_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=dim_embed),
            nn.Linear(in_features=dim_embed, out_features=num_classes))

    def forward(self, X):
        X = self.patch_embeddings(X)
        class_tokens = self.class_token.expand(X.shape[0], -1, -1)
        X = torch.cat((class_tokens, X), dim=1)
        X = self.dropout(X)

        for block in self.StackedTransformerEncoders:
            X = block(X)

        return self.classification_head(X[:, 0, :])