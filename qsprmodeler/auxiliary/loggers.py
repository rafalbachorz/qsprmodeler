import logging
        
def get_logger(logger_name):
    # create logger for
    if len(logging.getLogger(logger_name).handlers) == 0:
        log = logging.getLogger(logger_name)
        log.setLevel(level=logging.INFO)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # reate console handler for logger.
        ch = logging.StreamHandler()
        #ch.setLevel(level=logging.DEBUG)
        ch.setFormatter(formatter)
        log.addHandler(ch)
    else:
        log =  logging.getLogger(logger_name)   
    return log 