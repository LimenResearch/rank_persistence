import torch
import os
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from rank_persistence.utils import get_n_colors
from rank_persistence.applications.utils import load_dataset, generate_timestamp
from rank_persistence.applications.utils import train, test, BestModelSaver
from rank_persistence.applications.architectures import FC, Conv
from rank_persistence.applications.utils import get_activations_per_layer
from rank_persistence.applications.utils import plot_persistence_diagram
from rank_persistence.applications.utils import read_and_plot_tensorboard_json
from rank_persistence.applications.utils import get_accuracy_per_class
from rank_persistence.applications.dimensionality_reduction import EmbeddingVisualiser
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context="paper", style="white")


def train_and_test(model, device, train_loader, optimizer, saver, writer,
                   num_epochs = 20, loss_func = F.nll_loss):

    for epoch in range(1, num_epochs + 1):
        print("starting epoch {} of {}".format(epoch, num_epochs))
        train(model, device, train_loader, optimizer, epoch,
              loss_func = loss_func, writer = writer)
        loss, acc, data, labels, preds, target = test(model, device, test_loader,
                                                      epoch, writer = writer)
        saver.save(model, optimizer, epoch, loss, acc)

    return model, loss, acc, data, labels, preds, target

def min_max_scale(points):
    return (points - points.min()) / (points.max() - points.min())


def select_examples_per_class(points, labels, num_sample_per_class = 250):
    new_points = []
    new_labels = []

    for l in np.unique(labels):
        p = points[labels == l][:num_sample_per_class]
        new_points.extend(p)
        new_labels.extend(np.zeros(p.shape[0]) + l)

    new_points = np.array(new_points)
    new_labels = np.array(new_labels)
    np.random.seed(123)
    per = np.random.permutation(new_points.shape[0])
    return new_points[per], new_labels[per]

if __name__ == "__main__":
    plt.ion()
    device = torch.device("cuda")
    timestamp = generate_timestamp()
    writer = SummaryWriter()
    dataset_name = "MNIST"
    batch_size = 128
    data_folder = "./data"
    train_loader,\
    test_loader,\
    image_size = load_dataset(dataset_name, batch_size, data_folder)
    num_classes = 10
    conv_params = [[1, 4, 3], [4, 8, 3], [8, 16, 3]]
    hidden_sizes = [100]
    model = Conv(batch_size, image_size, conv_params, hidden_sizes,
                 num_classes, add_linear_out_per_layer = True)
    dummy_input = torch.rand((batch_size,) +  (image_size))
    writer.add_graph(model=model, input_to_model=(dummy_input, ))
    model = model.to(device)
    model_path = os.path.join('./checkpoints', timestamp)
    saver = BestModelSaver(model_path)
    optimizer = torch.optim.SGD(model.parameters(), lr= 5e-3)
    saver = BestModelSaver(model_path)
    model,\
    loss,\
    acc,\
    data,\
    labels, preds, target = train_and_test(model, device, train_loader, optimizer, saver,
                            writer, num_epochs = 2, loss_func = F.nll_loss)
    conf_mat = get_accuracy_per_class(num_classes, preds, target)

    # hidden_size = 100
    # model = FC(batch_size, image_size, hidden_size, num_classes)
    # load_path = None
    # # load_path = None
    # if load_path is not None:
    #     info_dict = torch.load(load_path)
    #     model.load_state_dict(info_dict['state_dict'])
    #     model = model.to(device)
    # else:
    #     dummy_input = torch.rand((batch_size,) +  (np.prod(image_size),))
    #     writer.add_graph(model=model, input_to_model=(dummy_input, ))
    #     model = model.to(device)
    #     optimizer = torch.optim.SGD(model.parameters(), lr= 5e-3)
    #     scheduler = StepLR(optimizer, step_size=1, gamma=.1)
    #     class_names = "_" + ''.join([str(c) for c in classes])
    #     model_path = os.path.join('./checkpoints', timestamp + class_names)
    #     saver = BestModelSaver(model_path)
    #     model = train_and_test(model, device, train_loader, optimizer, saver, writer,
    #                            num_epochs = 20, loss_func = F.nll_loss)
    # if classes is not None:
    #     num_colors = len(classes)
    # else:
    #     num_colors = np.unique(test_loader.dataset.dataset.targets).shape[0]
    # colors = get_n_colors(num_colors)
    # points, labels = get_activations_per_layer(model.model, test_loader,
    #                                              layer_num = -1)
    # points, labels  = select_examples_per_class(points, labels)
    # emb = EmbeddingVisualiser(labels, points, feature_names = ["activations"],
    #                           label_names = ["labels"],
    #                           session_name = ''.join([str(c) for c in classes]))
    # folder = os.path.join('./', 'embeddings')
    # emb.create_labels_file(folder)
    # emb.visualize(folder)
    #
    #
    # path = './tensorboard_json/state' + ''.join([str(c) for c in classes]) + "_l1.txt"
    # f, ax  = plt.subplots()
    # color_label_dict = {l : c for l, c in zip(classes, colors)}
    # read_and_plot_tensorboard_json(path, labels, proj_name = "umap",
    #                                    num_components=2, f=f, ax = ax,
    #                                    color_label_dict=color_label_dict)
