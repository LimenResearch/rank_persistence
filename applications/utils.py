import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import shutil
from datetime import datetime
import dionysus as d
from collections import Counter
from rank_persistence.persistence.persistence_diagram import PersistenceDiagram, CornerPoint
from rank_persistence.persistence.utils import __min_birth_max_death
from rank_persistence.utils import bottleneck
import matplotlib.pyplot as plt
import seaborn as sns
import json

sns.set(context="paper", style="white")

"""Pytorch data managing and wrappers
"""
class Conv2d_pad(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(Conv2d_pad, self).__init__(in_channels, out_channels, kernel_size,
                                         bias=False)
        self.equivaricance = 'translations'
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2)

    def __repr__(self):
        return "{}\nkernel size{}".format("Conv2d", self.kernel_size)


def generate_timestamp():
    return datetime.now().isoformat()[:-7].replace("T","-").replace(":","-")


def select_classes(dataset, classes):
    return torch.any(torch.stack([dataset.targets == c for c in classes]),
                     dim = 0).numpy()


def load_dataset(name, batch_size, data_folder = "./data", classes = None):
    if name == "FashionMNIST":
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])
        train_loader = torch.utils.data.DataLoader(
                            datasets.FashionMNIST(data_folder, train=True,
                                                  download=True, transform=transform),
                            drop_last=True, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
                            datasets.FashionMNIST(data_folder, train=False,
                                                  download=True, transform=transform),
                            drop_last=True, batch_size=batch_size, shuffle=True)
        image_size = (1,28,28)
    elif name == "MNIST":
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                        ])
        dataset = datasets.MNIST(data_folder, train=True, download=True,
                                 transform=transform)
        if classes is not None:
            inds = select_classes(dataset, classes)
            dataset = torch.utils.data.Subset(dataset, indices=np.where(inds == True)[0])
            train_loader = torch.utils.data.DataLoader(dataset,
                                drop_last=True, batch_size=batch_size, shuffle=True)
        else:
            train_loader = torch.utils.data.DataLoader(dataset,
                                drop_last=True, batch_size=batch_size, shuffle=True)


        dataset = datasets.MNIST(data_folder, train=False, download=True,
                                 transform=transform)
        if classes is not None:
            inds = select_classes(dataset, classes)
            dataset = torch.utils.data.Subset(dataset, np.where(inds == True)[0])
            test_loader = torch.utils.data.DataLoader(dataset,
                               drop_last=True, batch_size=batch_size, shuffle=True)
        else:
            test_loader = torch.utils.data.DataLoader(dataset,
                               drop_last=True, batch_size=batch_size, shuffle=True)
        image_size = (1,28,28)
    elif name == "CIFAR10":
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5),
                                                             (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root=data_folder, train=True,
                                                download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                   shuffle=True, drop_last=True)

        testset = torchvision.datasets.CIFAR10(root=data_folder, train=False,
                                               download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                  shuffle=False, drop_last=True)

        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        image_size = (3,32,32)
    else:
        raise NotImplementedError("The dataset is not available")
    return train_loader, test_loader, image_size


def train(model, device, train_loader, optimizer, epoch,
          loss_func = F.nll_loss, writer = None):
    steps = 150
    correct = 0
    n_total = 0
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        l1 = loss_func(output, target)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        n_total += data.shape[0]
        l1.backward()
        optimizer.step()
        if writer is not None:
            writer.add_scalar('Loss/train', l1.item(), epoch)
            writer.add_scalar('Acc/train', 100. * correct / n_total, epoch)
        if batch_idx % steps == 0 and batch_idx != 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc {:.3f}%'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), l1.item(),
                100. * correct / n_total))


def test(model, device, test_loader, epoch, loss_func = F.nll_loss,
         writer = None):
    model.eval()
    test_loss = 0
    correct = 0
    preds = []
    targets = []
    with torch.no_grad():

        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_func(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            preds.append(pred)
            targets.append(target)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_acc))
    if writer is not None:
        writer.add_scalar('Loss/val', test_loss, epoch)
        writer.add_scalar('Acc/val', test_acc, epoch)
    return test_loss, test_acc, data, target, torch.squeeze(torch.cat(preds)), torch.cat(targets)


class BestModelSaver:
    def __init__(self, path):
        self.loss = 1e+6
        self.check_path(path)
        self.path = os.path.join(path, 'checkpoint.pth.tar')
        self.best_path = os.path.join(path, 'model_best.pth.tar')

    @staticmethod
    def check_path(path):
        if not os.path.isdir(path):
            os.makedirs(path)


    def is_best(self, new_loss):
        return self.loss > new_loss


    def save(self, model, optimizer, epoch, loss, acc):
        state = {
            'epoch': epoch,
            'loss': loss,
            'state_dict': model.state_dict(),
            'best_acc1': acc,
            'optimizer' : optimizer.state_dict(),
        }
        torch.save(state, self.path)
        if self.is_best(loss):
            self.loss = loss
            shutil.copyfile(self.path, self.best_path)


def get_activations_per_layer(model, loader, device = torch.device("cuda"),
                              layer_num = None):

    if layer_num == -1:
        layer_num = len(model) -1

    model.eval()
    activations = []
    labels = []
    with torch.no_grad():

        for data, target in loader:
            input, target = data.to(device), target.to(device)
            if input.ndimension() == 4:
                input = input.view(loader.batch_size, -1)

            for i, layer in enumerate(model):
                input = layer(input)
                if layer_num is None or i == layer_num:
                    activations.extend(input)
                    labels.extend(target)

    activations = [a.cpu().detach().numpy() for a in activations]
    labels = [l.cpu().detach().numpy() for l in labels]
    return np.asarray(activations), np.asarray(labels)


def get_accuracy_per_class(num_classes, preds, target):
    confusion_matrix = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        for t, p in zip(target.view(-1), preds.view(-1)):
            print(t, p)
            confusion_matrix[t.long(), p.long()] += 1

    return confusion_matrix



"""Multicolored persistence
"""
def get_vietoris_rips_persistence_cycles(points, max_degree, max_distance, labels):
    filtration = d.fill_rips(points, max_degree, max_distance)
    filtration = sort_filtration_by_labels(filtration, labels)
    m = d.homology_persistence(filtration)
    return get_cycle_dict(m, filtration), filtration


def sort_filtration_by_labels(filtration, labels):
    labels_map = {i: l for i, l in enumerate(labels)}
    labelled_filtration = []
    ds = []

    for s in filtration:
        labelled_filtration.append([labels_map[v] for v in s])
        ds.append(s.data)

    ds = np.array(ds)
    uni_data = np.unique(ds)
    labelled_filtration = np.asarray(labelled_filtration)
    new_inds = []

    for d in uni_data:
        inds = np.where(ds == d)[0]
        if len(inds) == 1 or np.unique(ds[inds]) == 0:
            new_inds.extend(inds)
        else:
            els = labelled_filtration[inds]
            counted = [Counter(el) for el in els]
            sorting = np.argsort([max(c.values()) for c in counted])
            new_inds.extend(inds[sorting])

    filtration.rearrange(new_inds)
    return filtration


def get_cycle_dict(pers, f):
    return {i: ([sc.index for sc in c], f[pers.pair(i)].data, f[i].data) for i, c in enumerate(pers)
            if len(c)!= 0}


def get_labelled_cycles(cycle_dict, sorted_filtration, labels):
    cycle_labels = {}

    for key, value in cycle_dict.items():
        cycle_labels[key] = np.unique(np.concatenate([labels[list(sorted_filtration[simp_index])]
                             for simp_index in value[0]]))

    return cycle_labels


def combine_n_colors(colors):
    try:
        return tuple(np.asarray(colors).mean(axis=0))
    except:
        return colors


def build_dgms_from_dict(labelled_cycles_dict, colors, classes):
    degrees = np.unique([v[0] for v in labelled_cycles_dict.values()])
    cornerpoints = {d : [] for d in degrees}
    label_indices = {c: i for i,c in enumerate(classes)}

    for k, v in labelled_cycles_dict.items():
        color = combine_n_colors([colors[label_indices[l]] for l in v[1]])
        cp = CornerPoint(v[0], v[2], v[3], vertex = v[1], color=color)
        cornerpoints[v[0]].append(cp)

    dgms = [PersistenceDiagram(cornerpoints = cornerpoints[d])
                               for d in cornerpoints.keys()]
    return dgms


def plot_persistence_diagram(dgm, labels, alpha=0.6, min_persistence = 0,
                             ax = None):
    label_indices = {c: i for i,c in enumerate(labels)}
    cornerpoints = dgm.cornerpoints
    cornerpoints.sort(key=lambda x: x.persistence)
    persistence = [(c.k, (c.birth, c.death)) for c in dgm.cornerpoints
                   if c.persistence > min_persistence]
    (min_birth, max_death) = __min_birth_max_death(persistence, 0)
    ind = 0
    delta = ((max_death - min_birth) / 10.0)
    infinity = max_death + delta
    axis_start = min_birth - delta
    plt.plot([axis_start,infinity],[axis_start, infinity], color='k',
             alpha=alpha)
    plt.axhline(infinity,linewidth=1.0, color='k', alpha=alpha)
    plt.text(axis_start, infinity, r'$\infty$', color='k', alpha=alpha)
    if cornerpoints is not None:
        reversed_cornerpoints = reversed(cornerpoints)
    else:
        reversed_cornerpoints = [None] * len(persistence)

    # Draw points in loop
    used_labels = []

    for interval, cp in zip(reversed(persistence), reversed_cornerpoints):
        color = cp.color
        label = list(np.unique([labels[label_indices[v]] for v in cp.vertex]))
        if float(interval[1][1]) != float('inf'):
            plt.plot(interval[1][0], interval[1][1], alpha=alpha,
                        color = color,
                        label=str(label) if label not in used_labels else "",
                        marker='o')
            if label not in used_labels:
                used_labels.append(label)
            # plt.plot([interval[1][0],interval[1][0]],[interval[1][1], interval[1][0]],
            #          color = color, alpha = alpha/2,
            #          linestyle="dashed")
            # plt.plot([interval[1][0],interval[1][1]],[interval[1][1], interval[1][1]],
            #          color = color, alpha = alpha/2,
            #          linestyle="dashed")
        else:
            print("infinte death")
            print("interval: ", interval[1][0], infinity)
            plt.plot(interval[1][0], infinity, alpha=alpha,
                        color = color, label=str(label) if label not in used_labels else "",
                        marker='o')
            if label not in used_labels:
                used_labels.append(label)
            plt.plot([interval[1][0],interval[1][0]],[interval[1][0], infinity],
                     color = color, alpha = alpha)
        ind = ind + 1
    print("used labels ", used_labels)
    plt.title('Persistence diagram')
    plt.xlabel('Birth')
    plt.ylabel('Death')
    plt.legend()
    # Ends plot on infinity value and starts a little bit before min_birth
    plt.axis([axis_start, infinity, axis_start, infinity + delta])
    return plt


def load_diagram(path, degree = 1):
    cornerpoint_list = np.load(path, allow_pickle=True)
    return PersistenceDiagram(cornerpoints = cornerpoint_list[degree])


def get_unique_labels(dgm):
    labels = []

    for c in dgm.cornerpoints:
        if list(c.vertex) not in labels:
            labels.append(list(c.vertex))

    return sorted(labels)


def multicolored_bottleneck_distance(dgm1, dgm2, plot_subdiagrams = False,
                                     matching = False):
    labels1 = get_unique_labels(dgm1)
    labels2 = get_unique_labels(dgm2)

    if labels1 == labels2:
        print("measuring multicolored distance")
        labels = labels1
    else:
        labels = [l for l in labels1 if l in labels2]
        print("not all the colors are comparable " +
              "distance based on the subset {}".format(labels))
    if plot_subdiagrams:
        f, axs = plt.subplots(len(labels), 2)

    multicolored_distance = []
    if matching:
        multicolored_matching = []

    for i, l in enumerate(labels):
        print("computing distance for label {}".format(l))
        sub1 = PersistenceDiagram(cornerpoints = [c for c in dgm1.cornerpoints
                                                  if (list(c.vertex) == l and c.persistence !=0)])
        sub2 = PersistenceDiagram(cornerpoints = [c for c in dgm2.cornerpoints
                                                  if (list(c.vertex) == l and c.persistence !=0)])
        dist = bottleneck(sub1, sub2, proper = True, matching = matching)
        if matching:
            multicolored_distance.append(dist[0])
            multicolored_matching.append(dist[1])
        else:
            multicolored_distance.append(dist)
        if plot_subdiagrams:
            plot_persistence_diagram(dgm1, [l], ax =axs[i,0])
            plot_persistence_diagram(dgm2, [l], ax =axs[i,1])
    if matching:
        return multicolored_distance, multicolored_matching
    else:
        return multicolored_distance




"""Tensorboard plotting
"""
def read_and_plot_tensorboard_json(path, labels, proj_name = "umap",
                                   num_components=2, f = None, ax = None,
                                   color_label_dict=None):
    """Plot projections from tensorboard bookmark.

    Parameters
    ----------
    path : string
        Path to json/txt tensorboard bookmark file.
    labels : list
        Labels corresponding to the projected points.
    proj_name : string
        Analysis to plot: "umap", "pca", "tsne".
    num_components : int
        number of components
    ax : plt.ax
        figure axes matplotlib.pyplot ax
    color_label_dict : dict
        dictionary associating each label to a given color

    Returns
    -------
    np.ndarray
        projected points

    """
    with open(path) as json_file:
        data = json.load(json_file)

    projections = data[0]['projections']
    keys = [k for k in list(projections[0].keys())
            if proj_name in k][:num_components]
    projected_points = []

    for p in projections:
        projected_points.append([p[k] for k in keys])

    projected_points = np.asarray(projected_points)

    if ax is None:
        f, ax = plt.subplots()

    for l in np.unique(labels):
        scat = ax.scatter(projected_points[labels == l, 0],
                          projected_points[labels == l, 1],
                          color = color_label_dict[l], label = l)
        f.legend()

    return projected_points
