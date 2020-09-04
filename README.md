# Team Mongoose Handwritten Number Recogniser

Our reliance on electronic documents is increasing; however, the means of translating real-world information to a digital format is tedious and time-consuming. We want to develop an AI that could recognise a handwritten number, thus automating the conversion of real to digital data. As the result, we decided to implement our Handwritten Recognition so that it allows the handwritten images to be recognised. We implemented four neutral networks for our handwritten recognition.The four neutral network we implemented are LeNet5 Model, Mongoose Model, ResNet Model and Inception Model. In implementing those four models, we look at how effective are these model in recognising our handwritten digits through training and testing. 



### Prerequisites

Our models require pytorch and numpy to run.


Install Pytorch 

```
pip ~ install pytorch
```

Install Nympy

```
pip ~ install numpy
```


## Training the Algorithm

Choose your parameters by changing the variable values in the user interface.

To build a model create a instance of the builder class with your desired parameters
```
classifier = Builder(Algorithm, total_epoch, train_batch, learning_rate, momentum)
```

Run the builder classes train, load_data, load_model, test, evaluate or run through functions.

```
classifier.runthrough()
```



## Built With

* [PyTorch](https://pytorch.org/) - A deep learning framework


## Authors

* **Daniel Etzinger** - *Algorithms*
* **Kenneth Zhu** - *Algorithms, Test and Evaluation*


## Acknowledgments

* Thanks go to Yann LeCunn, creater of the MNIST database.
