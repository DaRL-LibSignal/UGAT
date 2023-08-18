import logging
from .task import BaseTask
from common.registry import Registry



@Registry.register_task("sim2real")
class SIM2REALTask(BaseTask):
    '''
    Register Traffic Signal Control task.
    '''
    def run(self):
        '''
        run
        Run the whole task, including training and testing.

        :param: None
        :return: None
        '''
        try:
            if Registry.mapping['model_mapping']['setting'].param['run_model']:
                self.trainer.run()

        except RuntimeError as e:
            self._process_error(e)
            raise e
