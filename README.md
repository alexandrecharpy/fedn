# FEDn

## What is FEDn?
FEDn is an open source framework for Federated Machine Learning (FedML), developed and maintained by Scaleout Systems. 

*Warning, this is an experimental version of the software. Do not use as is for production scenarios!*

## Why use FEDn? 

FEDn provides a flexible framework for building highly scalable federated machine learning systems using the ML framwork of your choice. It is designed to take FedML to production, paying equal attention to the requirements from developers of new models and algorithms and the operational perspective of running a production-grade distributed system. FEDn is based on open protocols and can be integrated with open MLOps frameworks like STACKn to provide end-to-end ML alliances.   

## Core Features
FEDn supports a highly scalable implementation of Federated Averaging. Other algorithms including meta-modeling is on the roadmap. 

<!--- #### Multimodal participation
#### Multilevel model combinations
---> 

## Getting started 

The easiest way to start with FEDn is to use the provided docker-compose templates to launch a sandbox environment consisting of the Controller, one Reducer, one Combiner, and a number of Clients. Together, this deploys a minimal system for training a federated model using the Federated Averaging strategy. This repository includes a number of test projects, as specified in this repository in the 'test' folder. These test projects can be used as templates for creating your own model code for the framework. 

Clone the repository and follow these steps: 

1. Create a file named '.env' in the repository root folder and set the following variables:
```yaml
EXAMPLE=mnist-multi-combiner
ALLIANCE_UID=ac435faef-c2df-442e-b349-7f633d3d5523
CLIENT_NAME_BASE=client-fedn1-
MDBUSR=
MDBPWD=
```
(Choose admin username and password for MongoDB)

> you set the EXAMPLE variable to the example you are working on imported with base path from test/your_example
or start all commands below by prepending ```EXAMPLE=mnist``` like ```$ EXAMPLE=data_center docker-compose up```

### Minimal standalone deployment 
We provide templates for a minimal standalone Docker deployment, useful for local testing and development. 

1. To deploy the server-side components (Controller, Minio, MongoDB and the Dashboard):

````bash 
$ docker-compose up 
````
Make sure you can reach the Controller on 'localhost:8080/controller' before proceeding to the next step. 

2. Attach a combiner:
````bash 
$ docker-compose -f combiner.yaml up 
````

3. Attach a number of Clients (assuming you are running the MNIST example):
````bash 
$ docker-compose -f mnist-clients.yaml up 
````
### Train a federated model

#### Seed the system with an initial model

Navigate to localhost:8080/controller, and navigate to the page for the deployed Combiner. There, you will find a configuration for the combiner's storage (Minio) as well as the option to configure a task for the combiner. Note the field "current_model". This is applied in this demo environment by a fixture seeding the database. You can find a pre-generated seed model in "test/mnist-multi-combiner/seed". To prepare FEDn to run training, we need to upload a corresponding seed model to the appropriate location in Minio. Navigate to localhost:9000 and log in (the credentials are available in the fixture file seed.yaml in in the root directory). Upload the seed model file in the bucket "models". 

*Note, the above instruction is assuming you are using the default development settings/naming conventions. You might need to adapt if you have altered the credentials for Minio, for example. There is also a file "init_model.py" that you can edit if you would like to alter the neural network itself.*

#### Start training
To start training, simply click the "Start" button from the Combiner page.  

## Where to go from here?
Explore our other example models, or use them as templates to create your own project. 

### Fork for FedBird

The original project has been forked in order to customize FEDn with
the FedBird project.  A new branch has been created: <fed_bird>.

To run the server and combiner, still using the standard docker-compose
commands:
- docker-compose up (for the "server")
- docker-compose -f combiner.yaml up (for the "combiner")
- COMPOSE_API_VERSION=1.40 docker-compose -f mnist-clients.yaml up (for the clients)

The latter is required if using the latest nvidia platform: Docker.io and docker-compose are not aligned.

If the latter command does work, you would need to patch docker compose:
- COMPOSE_API_VERSION=1.40 <pip install dir>/docker-compose -f mnist-clients.yaml up (for the clients)

as recommanded in the following link:
https://github.com/docker/compose/issues/6691


### Distributed deployment
Documentation coming soon. 

## Commercial support
Reach out to Scaleout to learn about how FEDn can be customized, configured and deployed to enable production-grade ML-alliances and life-cycle management of the federated models.  

## License
See LICENSE file.
