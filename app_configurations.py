import configparser

config = configparser.ConfigParser()
config.read('conf/application.conf')

"""
LOG Config
"""

LOG_LEVEL = config.get('LOG', 'log_level', fallback="INFO")
LOG_BASEPATH = config.get('LOG', 'base_path', fallback="logs/")
LOG_FILE_NAME = LOG_BASEPATH + config.get('LOG', 'file_name', fallback='connected-worker')
LOG_HANDLERS = config.get('LOG', 'handlers')
LOGGER_NAME = config.get('LOG', 'logger_name')