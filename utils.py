import matplotlib.pyplot as plt
def plot_scores(accuracy_scores):
    plt.plot(range(len(accuracy_scores)), accuracy_scores)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.ylim(ymin=0, ymax=1)
    plt.title('accuracy over epochs')
    plt.show()
