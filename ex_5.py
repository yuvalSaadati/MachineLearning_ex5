import os
import os.path
import soundfile as sf
import librosa
import numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optimizer
import torch.nn.functional as F

# gcommand_dataset.py file:
AUDIO_EXTENSIONS = [
    '.wav', '.WAV',
]
def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    spects = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_audio_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    spects.append(item)
    return spects


def spect_loader(path, window_size, window_stride, window, normalize, max_len=101):
    y, sr = sf.read(path)
    # n_fft = 4096
    n_fft = int(sr * window_size)
    win_length = n_fft
    hop_length = int(sr * window_stride)

    # STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)
    spect, phase = librosa.magphase(D)

    # S = log(S+1)
    spect = np.log1p(spect)

    # make all spects with the same dims
    # TODO: change that in the future
    if spect.shape[1] < max_len:
        pad = np.zeros((spect.shape[0], max_len - spect.shape[1]))
        spect = np.hstack((spect, pad))
    elif spect.shape[1] > max_len:
        spect = spect[:, :max_len]
    spect = np.resize(spect, (1, spect.shape[0], spect.shape[1]))
    spect = torch.FloatTensor(spect)

    # z-score normalization
    if normalize:
        mean = spect.mean()
        std = spect.std()
        if std != 0:
            spect.add_(-mean)
            spect.div_(std)

    return spect


class GCommandLoader(data.Dataset):
    """A google command data set loader where the wavs are arranged in this way: ::
        root/one/xxx.wav
        root/one/xxy.wav
        root/one/xxz.wav
        root/head/123.wav
        root/head/nsdf3.wav
        root/head/asd932_.wav
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        window_size: window size for the stft, default value is .02
        window_stride: window stride for the stft, default value is .01
        window_type: typye of window to extract the stft, default value is 'hamming'
        normalize: boolean, whether or not to normalize the spect to have zero mean and one std
        max_len: the maximum length of frames to use
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        spects (list): List of (spects path, class_index) tuples
        STFT parameter: window_size, window_stride, window_type, normalize
    """

    def __init__(self, root, transform=None, target_transform=None, window_size=.02,
                 window_stride=.01, window_type='hamming', normalize=True, max_len=101):
        classes, class_to_idx = find_classes(root)
        spects = make_dataset(root, class_to_idx)

        if len(spects) == 0:
            raise (RuntimeError(
                "Found 0 sound files in subfolders of: " + root + "Supported audio file extensions are: " + ",".join(
                    AUDIO_EXTENSIONS)))

        self.root = root
        self.spects = spects
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = spect_loader
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_type = window_type
        self.normalize = normalize
        self.max_len = max_len
        self.len = len(self.spects)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (spect, target) where target is class_index of the target class.
        """
        # print(index)
        path, target = self.spects[index]
        spect = self.loader(path, self.window_size, self.window_stride, self.window_type, self.normalize, self.max_len)
        # print (path)
        if self.transform is not None:
            spect = self.transform(spect)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return spect, target

    def __len__(self):
        return self.len


def _make_layers():
    """
    Create vgg layers:
    Convolution using 64 filters + Batch Normalization and relu + Max Pooling
    Convolution using 128 filters + Batch Normalization and relu + Max Pooling
    Convolution using 256 filters + Batch Normalization and relu
    Convolution using 256 filters + Batch Normalization and relu + Max Pooling
    Convolution using 512 filters + Batch Normalization and relu
    Convolution using 512 filters + Batch Normalization and relu + Max Pooling
    Convolution using 512 filters + Batch Normalization and relu
    Convolution using 512 filters + Batch Normalization and relu + Max Pooling
    Fully connected with 7680 nodes
    Fully connected with 512 nodes
    Output layer with Softmax activation with 30 nodes.
    :return: Sequential of all layers
    """
    cfg = [64, 'Max Pooling', 128, 'Max Pooling',
     256, 256, 'Max Pooling', 512, 512, 'Max Pooling', 512, 512, 'Max Pooling']
    model_layers = []
    in_channels = 1
    for x in cfg:
        if x == 'Max Pooling':
            model_layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            model_layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                       nn.BatchNorm2d(x),
                       nn.ReLU(inplace=True)]
            in_channels = x
    model_layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
    return nn.Sequential(*model_layers)


class VGGModel(nn.Module):
    """
    The class is inheriting from Module class and implement the Vgg model.
    """
    def __init__(self):
        super(VGGModel, self).__init__()
        self.features = _make_layers()
        self.fc1 = nn.Linear(7680, 512)
        self.fc2 = nn.Linear(512, 30)
        self.optimizer = optimizer.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return F.log_softmax(out, dim=1)


def train(train_loader, model):
    """
    Train the model with train set.
    :param train_loader: train set
    :param model: vgg model
    """
    num_epochs = 10
    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        model.train()
        correct = 0
        for i, (input_batch, labels) in enumerate(train_loader):
            # forward
            outputs = model(input_batch)
            loss = criterion(outputs, labels)
            # backward
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
            # predict
            _, predicted = torch.max(outputs.data, 1)
            # sum of the correct labels
            correct = (predicted == labels).sum().item()
            # if i % 50 == 0:
            #     print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%\n'.format(epoch + 1, num_epochs, i + 1,
            #                                                                         len(train_loader), loss.item(),
            #                                                                                     correct /(labels.size(0))))


def testValidation(train_loader_valid, model):
    """
    Test the model with validation set
    :param train_loader_valid: validation set
    :param model: vgg model
    :return: print test accuracy after running all the data on the model
    """
    # set thr model in eval mode
    model.eval()
    with torch.no_grad():
        # sum the correct predictions on each batch
        correct = 0
        for i, (input_batch, labels) in enumerate(train_loader_valid):
            outputs = model(input_batch)
            _, prediction = torch.max(outputs.data, 1)
            # sum up all correct predictions
            correct += (prediction == labels).sum().item()
        # thr amount of all the examples
        input_size = labels.size(0) * len(train_loader_valid)
        print('Test Accuracy:' + str((correct / input_size) * 100))



def test_to_file(test_set, test_loader, model, classes):
    """
    Run the model with test set and write to
    test_y file the predoctions of the model.
    :param test_set: GCommandLoader object that created from train set
    :param test_loader: test set in data loader
    :param model: vgg model
    :param classes: all 30 classes
    """
    model.eval()
    files_names = test_set.spects
    i = 0
    test_y = open('test_y', 'w')
    predictions_list = []
    for input_batch, _ in test_loader:
        output = model(input_batch)
        _, batch_predictions = torch.max(output.data, 1)
        batch_predictions = batch_predictions.tolist()

        for prediction in batch_predictions:
            prediction_name = classes[prediction]
            file_name = os.path.basename(files_names[i][0])
            predictions_list.append("{},{}\n".format(file_name, prediction_name))
            i += 1
    predictions_list = sorted(predictions_list, key=lambda x: int(x.split('.')[0]))
    for prediction in predictions_list:
        test_y.write(prediction)
    test_y.close()


def main():
    """
    In main function we load data,
    train the model and then test thr model.
    """
    # load data
    dataset = GCommandLoader('train')
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True, pin_memory=True)
    dataset_valid = GCommandLoader('valid')
    train_loader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=100, shuffle=False, pin_memory=True)
    dataset_test = GCommandLoader('./test')
    train_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=100, shuffle=False, pin_memory=True)

    # create model
    model = VGGModel()
    # 30 classes names
    classes = dataset.classes
    # train model
    train(train_loader, model)
    # test validation set on vgg model
    testValidation(train_loader_valid, model)
    # test a test set om vgg model
    test_to_file(dataset_test, train_loader_test, model, classes)


if __name__ == '__main__':
    main()