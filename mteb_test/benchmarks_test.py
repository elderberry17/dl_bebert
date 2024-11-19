import mteb
import yaml
from sentence_transformers import SentenceTransformer


if __name__ == "__main__":
    config_file_path = "mteb_test/mteb_test_config.yaml"
    with open(config_file_path, "r") as yamlfile:
        test_config = yaml.load(yamlfile, Loader=yaml.FullLoader)

    model_name = test_config['model_name']
    names = test_config['benchmark_names']
    model = SentenceTransformer(model_name)
    tasks = mteb.get_tasks(tasks=names)
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(model, output_folder=test_config['output_folder'])
    print('done!')