
## Parameter
#### Dataset Setting
`--dataset <dataset name>`

We can set ‘cifar10’, ‘cifar100’ , ‘fashin-mnist’ and 'emnist' for CIFAR-10, CIFAR-100, Fashion-MNIST and EMNIST.

`--model <model_name>`

We can set ‘resnet20’, ‘vgg’, and ‘cnn’ for ResNet-20, VGG-16, and CNN model.

`--num_classes <number>`

Set the number of classes Set 10 for CIFAR-10
Set 20 for CIFAR-100
Set 10 for Fashion-mnist
Set 26 for Emnist: --emnist_type letters
Set 47 for Emnist:--emnist_type bymerge
Set 62 for Emnist:--emnist_type byclass
`--num_channels <number>`

Set the number of channels of datasets.
Set 3 for CIFAR-10 and CIFAR-100. Set 1 for Fashion-MNIST and EMNIST.

#### Data heterogeneity
`--iid <0 or 1>`

0 – set non-iid 1 – set iid

`--data_beta <β>`

Set the β for the Dirichlet distribution

####  FL Settings
`--epochs <number of rounds>`

Set the number of training rounds.


#### Model setting
`-- algorithm <baseline name>`

* FedKDMR
* FedCodl
* FedMR
* FedAvg
* FedProx
* FedExP

`--KD_alpha <num>` 

Set the number of Distillation weight [0,1).

`-- first_stage_bound <num>`

Set the round number of the first stage for Pre-training


`--KD_buffer_bound <num>`
Set the round number of the first stage for KD buffer. Make sure '-- first_stage_bound' + '--KD_buffer_bound' < '--epochs' to achieve maximum distillation

