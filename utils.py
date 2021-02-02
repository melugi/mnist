import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader


def peek_into_dataloader(loader: DataLoader):
    images, labels = next(iter(loader))
    num_of_images = 60
    for index in range(1, num_of_images + 1):
        plt.subplot(6, 10, index)
        plt.axis('off')
        plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')

    plt.show()

