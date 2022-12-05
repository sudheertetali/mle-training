
## Procedure to run and install code
Go to project root directory assignment_3.1/assign_sudheer
   
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
3->you can import package "assign_sudheer" by going to dist-> by running 
pip install assign_sudheer-0.0.1-py3-none-any.whl (wheel) file

    then you can run all files by running mlflowrun.py

4->
test files are made :
    instalation_test.py
    installTest.py
    test_1.py

5->
ingest_data.py
train.py
score.py
 can be found in assign_sudheer which can be run by running 
 python mlflowrun.py

 6->test files added

 7->
 html files have been generated using sphinix
 which can be found in docs/build/html