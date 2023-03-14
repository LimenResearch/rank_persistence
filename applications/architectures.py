import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from rank_persistence.applications.utils import Conv2d_pad

class FC(nn.Module):
    def __init__(self, batch_size, input_size, hidden_size, num_classes):
        super(FC, self).__init__()
        self.batch_size = batch_size
        if len(input_size) == 3:
            print("flattening input size")
            input_size = np.prod(input_size)
        self.model = nn.ModuleList([nn.Linear(input_size, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, num_classes),
                                    nn.ReLU()])

    def forward(self, x):
        if x.ndimension() == 4:
            x = x.view(self.batch_size, -1)
        for l in self.model:
            x = l(x)
        return  F.log_softmax(self.model(x), dim=1)



class Conv(nn.Module):
    def __init__(self, batch_size, input_size, conv_params, hidden_sizes,
                 num_classes, add_linear_out_per_layer = False):
        """conv_params: list of tuples (in_channels, out_channels, kernel_size)
        input_size is num_ch, width, height
        """
        super(Conv, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.conv_params = conv_params
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.add_linear_out = add_linear_out_per_layer
        self.build()


    def build(self):
        self.conv_module = nn.ModuleList([])
        self.classifier = nn.ModuleList([])
        self.linear_out = nn.ModuleList([])
        temp_size = np.array(self.input_size)[1:]

        for in_c, out_c, k_size in self.conv_params:
            self.conv_module.append(Conv2d_pad(in_c, out_c, k_size))
            self.conv_module.append(nn.MaxPool2d(2))
            temp_size = temp_size // 2
            if self.add_linear_out:
                self.linear_out.append(nn.Linear(out_c * np.prod(temp_size),
                                                 self.num_classes))


        for h_size in self.hidden_sizes:
            self.classifier.append(nn.Linear(out_c * np.prod(temp_size), h_size))
            temp_size = h_size
            self.classifier.append(nn.ReLU())
            if self.add_linear_out:
                self.linear_out.append(nn.Linear(h_size, self.num_classes))

        self.classifier.append(nn.Linear(h_size, self.num_classes))


    def forward(self, input):
        x = input
        for l in self.conv_module: x = l(x)
        x = x.view(self.batch_size, -1)
        for f in self.classifier: x = f(x)
        return  F.log_softmax(x, dim=1)


    def get_intermidiate_output(self, input, layer = -1):
        if layer == -1:
            return self.forward(input)
        else:
            x = input
            linear_out = self.linear_out[layer]
            if layer <= len(self.conv_module):
                for i in range(layer): x = self.conv_module[i][x]
            else:
                layer = layer - len(self.conv_module)
                for l in self.conv_module: x = l(x)
                for i in range(layer): x = self.classifier[i](x)
            return F.log_softmax(linear_out(x), dim=1)
