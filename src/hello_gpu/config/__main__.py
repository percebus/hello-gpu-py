from src.hello_gpu.config.configuration import configuration

logger = configuration.get_logger()


def run() -> None:
    print(configuration.settings)


if __name__ == "__main__":
    run()
