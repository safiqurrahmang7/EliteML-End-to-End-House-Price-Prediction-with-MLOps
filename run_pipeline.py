import click
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from pipelines.Training_pipelines import ml_pipepline
import logging


@click.command
def main():

    run = ml_pipepline()


    logging.info(
        "Now run \n "
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the experiment."
    )



if __name__ == '__main__':
    main()