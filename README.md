
## Procedure to run 
Go to project root directory
   
First initialize the mlflow by running <mlflow ui> in terminal
1) run the script mlflowrun.py to run the entire ml scripts.
2)To run python script  run python < scriptname.py >
3) results will get displayed in the terminal and all the metrics,parameters,artifacts gets logged into mlflow.

1->updated readme.md


2->added env.yaml file 
    -> Create all required dependencies from env.yml file using the below code
       #### Command to create an environment from the env.yml file
       conda env create --prefix ./env --file environment.yml
    -> Then activate the environment and select the python interpreter path
         ## activate environment
            conda activate < environment name > 
