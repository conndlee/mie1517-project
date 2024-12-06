

from video_lib import *


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    classifier = CNNClassifierAlex()
    state = torch.load("./model_CNNA_bs128_lr0.001_epoch10", map_location=device, weights_only=True)
    classifier.load_state_dict(state)
    model = CNNA2()
    model.classifier = classifier
    model.eval()
    model.to(device)
    classes = ["glasses", "no glasses", "safety glasses"]
    n = len(classes)
    activation_images = []
    titles = []
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    fig.suptitle("Neuron Maximum Activation Images")
    torch.manual_seed(1001)
    for i in range(n):
        tensor = gradient_ascent_max(model, i, input_shape=(1,3,224,224),steps=500, lr=0.1,regularization=True)
        image = tensor.cpu().squeeze().detach().numpy().transpose(1, 2, 0)
        image -= image.min()    
        image /= image.max()
        activation_images.append(image)
        titles.append(f"{classes[i]}")

    for i, (img, title) in enumerate(zip(activation_images, titles)):
        ax = axes[i] if n > 1 else axes 
        ax.imshow(img)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    plt.show()
    return 0

if __name__ == "__main__":
    main()