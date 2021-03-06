"""
    С помощью класса ISearch реализуется метод
    автоматического подбора гиперпараметров ISearch.
    Предполагается, что ISearch используется совместно с классом Engine,
    отвечающим за интерактивность метода
    ISearch может получать на вход результаты обучения и
    по этим результатом возврящать предлагаемую следующую точку для обучения.
    При необходимости ISearch может объяснить причины своего выбора.
    Пользователь не обязан следовать рекомендациям алгоритма.
    В ISearch можно передавать не только результаты обучения для тех точек,
    которые предложил алгоритм.
    Можно передавать ему любые результаты обучения,
    которые пользователь посчитает нужным получить
"""
import random
import copy
import numpy as np
import matplotlib.pyplot as plt


class ISearch():
    """
    Реализован класс отвечающий за выбор следующей точки для обучения.
    ISearch получает точку и результаты ее обучения через функцию add_result.
    По полученным данным ISearch решает
    какие гиперпараметры стоит обучить следующими.
    Запросить гиперпараметры для обучения можно функцией newt_point
    Запросить причины выбора следующей точки для обучения можно функцией why
    """
    def __init__(self):
        self.best_model = dict()
        self.best_point = dict()
        self.overfit_model = dict()
        self.overfit2_model = dict()
        self.overfit_criterion = ""
        self.overfit_point = dict()
        self.overfit2_point = dict()
        self.process_flag = "lr"
        self.prev_process = "lr"
        self.random_accuracy = 10
        self.added_model = dict()
        self.added_point = dict()
        self.warm_up_start = 0
        # for wd variables
        self.patience_wd = 0.000001
        self.left_wd_cur = 3
        self.left_wd = 3
        self.default_wd = 0
        self.right_wd = 7
        self.right_wd_cur = 7
        self.wd1 = 0
        self.wd1_deg = 0
        self.wd2_deg = 0
        self.res1 = 0
        self.wd2 = 0
        self.res2 = 0
        self.flag_wd = -1
        # variables for LR search
        self.default_epochs = 1
        self.start_lr = 0.1
        self.found_lr = -1
        self.left_lr = 0
        self.right_lr = self.start_lr
        self.patience_lr = 4
        self.cur_pat = -1
        self.decreasing = True
        # variables for random search
        self.patience_right = 30
        self.patience_left = 0
        self.warm_up_left = 0
        self.warm_up_right = 30
        self.threshold_left = 0
        self.threshold_right = 4
        self.factor_left = 0.046
        self.factor_right = 2
        self.momentum_left = 0
        self.momentum_right = 1
        self.suggest_point = {
            "lr": self.start_lr,
            "wd": self.default_wd,
            "patience": random.randint(
                self.patience_left, self.patience_right),
            "warm_up": 0,
            "threshold": 10**-random.uniform(
                self.threshold_left, self.threshold_right
            ),
            "parameter": "test_loss",
            "mode": "min",
            "factor": 10**-random.uniform(self.factor_left, self.factor_right),
            "start_lr": self.warm_up_start,
            "momentum": 0.9,
            "flag": "default",
            "epochs": self.default_epochs,
            "id": 0
        }

    def add_result(self, data, point):
        """
        Функция принимает на вход статистики обучения и
        точку для которой они были получены
        """
        self.added_model = data
        self.added_point = point

    def next_point(self):
        """
        Функция которая возвращает предлагаемую
        алгоритмом следующую точку для обучения.
        Имеет смысл вызывать после функции add_result,
        если это не начало работы программы.
        Если добавить точку не предлагаемую
        алгоритмом начнется случайный поиск
        """
        if self.added_point != {} and self.added_point != self.suggest_point:
            self.process_flag = "random"
            self.prev_process = "random"
        if self.process_flag == "lr":
            if self.decreasing is True:
                if self.cur_pat == -1:
                    self.suggest_point["id"] = 0
                    self.cur_pat = 1 % self.patience_lr
                    return copy.deepcopy(self.suggest_point), self.stop_signal
                if self.cur_pat == 0:
                    if (self.added_model["data"]["accuracy"] >
                            self.random_accuracy):
                        self.right_lr *= 2
                        self.suggest_point["lr"] = self.right_lr
                        self.suggest_point["start_lr"] = self.suggest_point[
                            "lr"
                        ]
                        self.cur_pat = (self.cur_pat+1) % self.patience_lr
                        self.suggest_point["id"] = self.cur_pat
                        return copy.deepcopy(
                            self.suggest_point), self.stop_signal
                    else:
                        self.decreasing = False
                        self.cur_pat = 0
                        self.left_lr = round(self.right_lr/2, 1)
                else:
                    if (self.added_model["data"]["accuracy"] >
                            self.random_accuracy):
                        self.suggest_point["id"] = self.cur_pat
                        self.cur_pat = (self.cur_pat+1) % self.patience_lr
                        return copy.deepcopy(
                            self.suggest_point), self.stop_signal
                    else:
                        self.decreasing = False
                        self.cur_pat = 0
                        self.left_lr = round(self.right_lr/2, 1)

            if self.decreasing is False:
                if (self.cur_pat == 0 and
                        round(self.right_lr-self.left_lr, 1) == 0.1):
                    self._lr_found()
                else:
                    if self.cur_pat == -1:
                        self.suggest_point["lr"] = round(
                            (self.left_lr+self.right_lr)/2, 1
                        )
                        self.suggest_point["start_lr"] = self.suggest_point[
                            "lr"
                        ]
                        self.suggest_point["id"] = 0
                        self.cur_pat = 1 % self.patience_lr
                        return copy.deepcopy(
                            self.suggest_point), self.stop_signal
                    elif self.cur_pat != 0:
                        if (self.added_model["data"]["accuracy"] >
                                self.random_accuracy):
                            self.suggest_point["id"] = self.cur_pat
                            self.cur_pat = (self.cur_pat+1) % self.patience_lr
                            return copy.deepcopy(
                                self.suggest_point), self.stop_signal
                        else:
                            self.cur_pat = 0
                    if (self.cur_pat == 0 and
                            round(self.right_lr-self.left_lr, 1) == 0.1):
                        self._lr_found()
                    elif self.cur_pat == 0:
                        if (self.added_model["data"]["accuracy"] >
                                self.random_accuracy):
                            self.left_lr = self.suggest_point["lr"]
                        else:
                            self.right_lr = self.suggest_point["lr"]
                        if round(self.right_lr-self.left_lr, 1) == 0.1:
                            self._lr_found()
                        else:
                            self.suggest_point["lr"] = round(
                                (self.left_lr+self.right_lr)/2, 1
                            )
                            self.suggest_point[
                                "start_lr"
                            ] = self.suggest_point[
                                "lr"
                            ]
                            self.suggest_point["id"] = self.cur_pat
                            self.cur_pat = (self.cur_pat+1) % self.patience_lr
                            return copy.deepcopy(
                                self.suggest_point), self.stop_signal

        if self.process_flag == "wd":
            if (self.added_model["data"]["accuracy"] >
                    self.best_model["data"]["accuracy"]):
                self.best_point = self.added_point
                self.best_model = self.added_model
            if self.flag_wd == 2:
                self.res2 = self.added_model["data"]["accuracy"]
                self.flag_wd = 0
                if abs(self.wd1-self.wd2) < self.patience_wd:
                    self.process_flag = "random"
                    self.prev_process = "wd"
                    self.wd1 = 0
                    self.wd1_deg = 0
                    self.res1 = 0
                    self.right_wd_cur = self.right_wd
                    self.left_wd_cur = self.left_wd
                    self.wd2 = 0
                    self.wd2_deg = 0
                    self.res2 = 0
                    self.flag_wd = -1
                else:
                    if self.res1 > self.res2:
                        self.right_wd_cur = self.wd2_deg
                    if self.res2 > self.res1:
                        self.left_wd_cur = self.wd1_deg
                    if self.res1 == self.res2:
                        self.right_wd_cur = self.wd2_deg
                        self.left_wd_cur = self.wd1_deg
            if self.flag_wd == 0:
                self._new_wd()
                return copy.deepcopy(self.suggest_point), self.stop_signal
            if self.flag_wd == 1:
                self.res1 = self.added_model["data"]["accuracy"]
                self.flag_wd = 2
                self.suggest_point["wd"] = self.wd2
                self.suggest_point["flag"] = "EarlyStopping"
                return copy.deepcopy(self.suggest_point), self.stop_signal

        if self.process_flag == "random":
            overfit = False
            if self.prev_process == "wd":
                self.prev_process = "random"
            elif self.best_model != {}:
                overfit, criterion = self._overfitting()
            if overfit is True:
                self.prev_process = "random"
                if criterion == "best":
                    self.overfit_model = self.best_model
                    self.overfit_point = self.best_point
                    self.overfit2_model = self.added_model
                    self.overfit2_point = self.added_point
                    self.suggest_point = self.best_point
                    if (self.best_model["data"]["accuracy"] <
                            self.added_model["data"]["accuracy"]):
                        self.overfit_criterion = "_accuracy"
                        self.best_point = self.added_point
                        self.best_model = self.added_model
                    else:
                        self.overfit_criterion = "_loss"
                    self._new_wd()
                    return copy.deepcopy(
                        self.suggest_point), self.stop_signal
                else:
                    self.overfit_model = self.added_model
                    self.overfit_point = self.added_point
                    self.overfit2_model = self.best_model
                    self.overfit2_point = self.best_point
                    self.suggest_point = self.added_point
                    if (self.best_model["data"]["accuracy"] <
                            self.added_model["data"]["accuracy"]):
                        self.overfit_criterion = "_loss"
                        self.best_point = self.added_point
                        self.best_model = self.added_model
                    else:
                        self.overfit_criterion = "_accuracy"
                    self._new_wd()
                    return copy.deepcopy(
                        self.suggest_point), self.stop_signal
            if overfit is False:
                if self.best_model != {}:
                    if (self.best_model["data"]["accuracy"] <
                            self.added_model["data"]["accuracy"]):
                        self.best_point = self.added_point
                        self.best_model = self.added_model
                else:
                    self.best_point = self.added_point
                    self.best_model = self.added_model
                parameter = random.choice(["test_loss", "test_accuracy"])
                mode = ""
                if parameter == "test_loss":
                    mode = "min"
                else:
                    mode = "max"
                if self.found_lr == -1:
                    learning_rate = round(random.uniform(0.1, 2), 5)
                else:
                    learning_rate = self.found_lr
                self.suggest_point = {
                    "lr": learning_rate,
                    "wd": self.default_wd,
                    "patience": random.randint(
                        self.patience_left, self.patience_right),
                    "warm_up": random.randint(
                        self.warm_up_left, self.warm_up_right),
                    "threshold": 10**-random.uniform(
                        self.threshold_left, self.threshold_right),
                    "parameter": parameter,
                    "mode": mode,
                    "factor": 10**-random.uniform(
                        self.factor_left, self.factor_right),
                    "start_lr": self.warm_up_start,
                    "momentum": 0.9,
                    "flag": "EarlyStopping",
                    "epochs": self.default_epochs,
                    "id": 0
                }
                return copy.deepcopy(self.suggest_point), self.stop_signal

    def why(self, flag=False):
        """
        Функция, объясняющая причины выбора следующей точки для обучения.
        В случае поиска weight dacay также выводит графики демонстрирующие
        найденное переобучение
        """
        if self.process_flag == "lr":
            return "Triyng to find the largest possible lr"
        if self.process_flag == "wd":
            if flag is not False:
                plots = [
                    "test" + self.overfit_criterion,
                    "train" + self.overfit_criterion
                    ]
                datas = [
                    self.overfit_model["data"],
                    self.overfit2_model["data"]
                    ]
                points = [str(self.overfit_point), str(self.overfit2_point)]
                fig, (axes1, axes2) = plt.subplots(1, 2, figsize=(20, 8))
                for j, data in enumerate(datas):
                    axes1.plot(data["x"], data[plots[0]], label=points[j])
                for j, data in enumerate(datas):
                    axes2.plot(data["x"], data[plots[1]], label=points[j])
                handles, labels = axes1.get_legend_handles_labels()
                plt.legend(
                    handles,
                    labels,
                    loc='upper center',
                    bbox_to_anchor=(-0.1, -0.2)
                )
                fig.suptitle("demonstration of overfitting")
                axes1.grid('on')
                axes1.set_xlabel('iteration')
                axes1.set_ylabel(plots[0])
                axes2.grid('on')
                axes2.set_xlabel('iteration')
                axes2.set_ylabel(plots[1])
                plt.show()
            return ("Trying to reduce overfitting. " +
                    "Trying to find wd by ternary search. " +
                    "You can also use data augumentation")
        if self.process_flag == "random":
            return ("Using The largest possible LR. " +
                    "Default wd = " + str(self.default_wd) +
                    ". Other parameters are randomized")

    def stop_signal(self, data, flag, patience, parameter, epochs):
        """
        callback функция, передаваемая в пользовательскую
        функцию для обучения сети.
        stop_signal отвечает за остановку обучения в ползовательской функции.
        Обучение останавливается либо через заданное колличество эпох,
        либо реализуется early stopping.
        flag передает информацию о том по какому
        критерию функция будет решать о том,
        что пора остановить обучение.
        epochs передает колличество эпох, которое должна обучится сеть
        (если flag == "default").
        data передает статистики обучения для реализации ранней остановки
        parameter паредает информацию о том, по какому критерию
        реализовывать раннюю остановку (test_accuracy, test_loss).
        patience передает информацию о том, сколько ждать
        улучшения перед тем, как остановиться.
        """
        if flag == "default":
            if len(data["test_loss"]) == epochs:
                return True
            else:
                return False
        if flag == "EarlyStopping":
            if parameter == "test_accuracy":
                if (len(data["test_accuracy"]) -
                        np.argmax(data["test_accuracy"]) > patience):
                    return True
                else:
                    return False
            elif parameter == "test_loss":
                if (len(data["test_loss"]) -
                        np.argmin(data["test_loss"]) > patience):
                    return True
                else:
                    return False

    def _new_wd(self):
        self.process_flag = "wd"
        self.wd1_deg = self.left_wd_cur + (
            self.right_wd_cur - self.left_wd_cur
        )/3
        self.wd2_deg = self.right_wd_cur - (
            self.right_wd_cur - self.left_wd_cur
        )/3
        self.flag_wd = 1
        self.wd1 = 10**-self.wd1_deg
        self.wd2 = 10**-self.wd2_deg
        self.suggest_point["wd"] = self.wd1
        self.suggest_point["flag"] = "EarlyStopping"

    def _lr_found(self):
        self.process_flag = "random"
        self.prev_process = "lr"
        self.found_lr = self.left_lr
        self.suggest_point["lr"] = self.found_lr
        self.suggest_point["start_lr"] = self.suggest_point["lr"]

    def _overfitting(self):
        average_best_test = 0
        average_best_train = 0
        average_cur_test = 0
        average_cur_train = 0
        amount = 10
        criterion = "_accuracy"
        average_best_test = sum(
            self.best_model["data"]["test"+criterion][
                max(0, len(
                    self.best_model["data"]["test"+criterion])-amount
                    ):len(self.best_model["data"]["test"+criterion])-1
            ]
        )/amount
        average_best_train = sum(
            self.best_model["data"]["train"+criterion][
                max(0, len(
                    self.best_model["data"]["train"+criterion])-amount
                    ):len(self.best_model["data"]["train"+criterion])-1
            ]
        )/amount
        average_cur_test = sum(
            self.added_model["data"]["test"+criterion][
                max(0, len(
                    self.added_model["data"]["test"+criterion])-amount
                    ):len(self.added_model["data"]["test"+criterion])-1
            ]
        )/amount
        average_cur_train = sum(
            self.added_model["data"]["train"+criterion][
                max(0, len(
                    self.added_model["data"]["train"+criterion])-amount
                    ):len(self.added_model["data"]["train"+criterion])-1
            ]
        )/amount
        if (average_best_test > average_cur_test and
                average_cur_train > average_best_train):
            return True, "cur"
        if (average_best_test < average_cur_test and
                average_cur_train < average_best_train):
            return True, "best"
        criterion = "_loss"
        average_best_test = sum(
            self.best_model["data"]["test"+criterion][
                max(0, len(
                    self.best_model["data"]["test"+criterion])-amount
                    ):len(self.best_model["data"]["test"+criterion])-1
            ]
        )/amount
        average_best_train = sum(
            self.best_model["data"]["train"+criterion][
                max(0, len(
                    self.best_model["data"]["train"+criterion])-amount
                    ):len(self.best_model["data"]["train"+criterion])-1
            ]
        )/amount
        average_cur_test = sum(
            self.added_model["data"]["test"+criterion][
                max(0, len(
                    self.added_model["data"]["test"+criterion])-amount
                    ):len(self.added_model["data"]["test"+criterion])-1
            ]
        )/amount
        average_cur_train = sum(
            self.added_model["data"]["train"+criterion][
                max(0, len(
                    self.added_model["data"]["train"+criterion])-amount
                    ):len(self.added_model["data"]["train"+criterion])-1
            ]
        )/amount
        if (average_best_test < average_cur_test and
                average_cur_train < average_best_train):
            return True, "cur"
        if (average_best_test > average_cur_test and
                average_cur_train > average_best_train):
            return True, "best"
        return False, "none"
