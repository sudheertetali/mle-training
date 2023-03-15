
## Procedure to run and install code
Go to project root directory assignment_4.1/tamlep_4_1_tetali_sudheer

a)->you can import package "tamlep_4_1_tetali_sudheer" by running below code in the terminal
pip install -i https://test.pypi.org/simple/ tamlep_4_1_tetali_sudheer==0.0.2 -t directory_path



First initialize the mlflow by running <mlflow ui> in terminal
c) ->added env.yaml file
    -> Create all required dependencies from env.yml file using the below code
       #### Command to create an environment from the env.yml file
       conda env create --prefix ./env --file env.yml
    -> Then activate the environment and select the python interpreter path
         ## activate environment
            conda activate < environment name >
1) run the script mlflowrun.py to run the entire ml scripts.
2)To run python script  run python < scriptname.py >
3) results will get displayed in the terminal and all the metrics,parameters,artifacts gets logged into mlflow.

b)->updated readme.md



d) ->test the installation by running the test files
test files are made :
    installTest.py
    unittest_1.py