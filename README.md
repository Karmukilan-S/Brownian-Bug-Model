# Brownian-Bug-Model
This contains python code for simulating Brownian Bug(BB), Active Brownian Bug model and it's variants

GOAL: To understand the role of replication on the population structure. 

This will be accomplished by contrasting two classes of proliferating Active Brownian Bugs; One with a constant replication rate and other with a density 
dependent replication rate(captures competition)

##Workflow
- In order to understand the role of replication better, we will strip away the structure and look at the population evolution of the model.The code is [here](https://github.com/Karmukilan-S/Brownian-Bug-Model/blob/main/Brownian%20motion%20animation)
- Now we are in a better spot to appreciate the spatial clustering observed in the proliferating brownian bug model(with periodic boundary condition and advection)
  The code is [here](https://github.com/Karmukilan-S/Brownian-Bug-Model/blob/main/Brownian%20model%20with%20proliferation%20and%20convection)
- One might wonder if changing the periodic boundary condition to a reflective one might yield in a different evolution. Though the exact evolution will be different
  the patterns observed remains unchanged. The code to the reflective model is [here](https://github.com/Karmukilan-S/Brownian-Bug-Model/blob/main/BB%20with%20proliferation(reflective%20boundary%20conditions))
- Let's add an active component to this model we developed to get our Constant replication rate ABB.The code for this model can be found [here](https://github.com/Karmukilan-S/Brownian-Bug-Model/blob/main/Active%20Brownian%20Bug%20model)
- Now we are in a great position to compare the patterns observed in both the model.The code is [here](https://github.com/Karmukilan-S/Brownian-Bug-Model/blob/main/Main)

This work is my take on understanding the SURF proposal(SURF conducted by IFISC Spain)"A simple model of active reproducing matter"


