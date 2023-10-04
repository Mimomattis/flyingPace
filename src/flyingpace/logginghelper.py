import getpass  # for getpass.getuser()
import logging
import socket

def init_logger():
    '''Returns a logger with predefined formatting'''
    hostname = socket.gethostname()
    username = getpass.getuser()
    LOG_FMT = '%(asctime)s %(levelname).1s - %(message)s'.format(hostname)
    formatter = logging.Formatter(LOG_FMT)
    logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%Y/%m/%d %H:%M:%S")
    log = logging.getLogger()
    log_file_name = 'log.txt'
    log.info("Redirecting log into file {}".format(log_file_name))
    fileh = logging.FileHandler(log_file_name, 'a')
    fileh.setFormatter(formatter)
    log.addHandler(fileh)

    return log

def log_output_and_errors(func, logger):
    '''Alters the Connection.run function to redirect fabric output to the logger'''
    def wrapper_with_logging(self, command, hide=None, **kwargs):
        with self.prefix("source ~/.bashrc"):
            result = func(self, command, hide=hide, **kwargs)
            output_buffer = ''
            error_buffer = ''

            def log_buffer(buffer, log_func):
                lines = buffer.split('\n')
                for line in lines:
                    if line.strip():
                        log_func(line.strip())

            for char in result.stdout:
                if char == '\n':
                    log_buffer(output_buffer, logger.info)
                    output_buffer = ''
                else:
                    output_buffer += char

            for char in result.stderr:
                if char == '\n':
                    log_buffer(error_buffer, logger.error)
                    error_buffer = ''
                else:
                    error_buffer += char

            log_buffer(output_buffer, logger.info)
            log_buffer(error_buffer, logger.error)

            return result
    return wrapper_with_logging