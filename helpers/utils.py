def print_losses_to_file(file_logger, step, training_losses, testing_losses):
    if len(training_losses) == 0 or len(testing_losses) == 0:
        return
    file_logger.write([step, training_losses[-1], testing_losses[-1]])
