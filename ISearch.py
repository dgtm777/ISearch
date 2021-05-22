class ISearch():
  def __init__(self):
    self.best_model = dict()
    self.best_point = dict()
    self.process_flag = "lr"
    self.prev_process = "lr"
    self.random_accuracy = 10
    self.added_model = dict()
    self.added_point = dict()
    self.warm_up_start = 0
    #for wd variables
    self.patience_wd = 0.000001
    self.left_wd = 3
    self.default_wd = 0
    self.right_wd = 7
    self.wd1 = 0
    self.wd1_deg = 0
    self.wd2_deg = 0
    self.res1 = 0
    self.wd2 = 0
    self.res2 = 0
    self.flag_wd = -1
    #variables for LR search
    self.default_epochs = 1
    self.start_lr = 0.1
    self.found_lr = -1
    self.left_lr = 0
    self.right_lr = self.start_lr
    self.patience_lr = 4
    self.cur_pat = -1
    self.decreasing = True
    import random
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
      "patience": random.randint(self.patience_left, self.patience_right),
      "warm_up": 0,
      "threshold": 10**-random.uniform(self.threshold_left, self.threshold_right), 
      "parameter": "test_loss",
      "mode": "min",
      "factor": 10**-random.uniform(self.factor_left, self.factor_right),
      "start_lr": self.warm_up_start,
      "momentum": 0.9,
      "flag": "default",
      "epochs": self.default_epochs,
      "id": 0
    }
  def addResult(self, data, point):
    self.added_model = data
    self.added_point = point
  def nextPoint(self):
    import copy
    if self.added_point != {} and self.added_point != self.suggest_point:
      self.process_flag = "random"
      self.prev_process = "random"
    if self.process_flag == "lr":
      if self.decreasing == True:
        if self.cur_pat == -1:
          self.suggest_point["id"] = 0
          self.cur_pat = 1%self.patience_lr
          return copy.deepcopy(self.suggest_point), self.stopSignal
        if self.cur_pat == 0:
          if self.added_model["data"]["accuracy"] > self.random_accuracy:
            self.right_lr *= 2     
            self.suggest_point["lr"] = self.right_lr    
            self.suggest_point["start_lr"] = self.suggest_point["lr"] 
            self.cur_pat = (self.cur_pat+1)%self.patience_lr
            self.suggest_point["id"] = self.cur_pat  
            return copy.deepcopy(self.suggest_point), self.stopSignal
          else:
            self.decreasing = False
            self.cur_pat = 0
            self.left_lr = round(self.right_lr/2, 1)
        else:
          if self.added_model["data"]["accuracy"] > self.random_accuracy:
            self.suggest_point["id"] = self.cur_pat
            self.cur_pat = (self.cur_pat+1)%self.patience_lr
            return copy.deepcopy(self.suggest_point), self.stopSignal
          else:
            self.decreasing = False
            cutPat = 0
            self.left_lr = round(self.right_lr/2, 1)

      if self.decreasing == False:
        if self.cur_pat == 0 and round(self.right_lr-self.left_lr, 1) == 0.1:
          self._lrFound()
        else:
          if self.cur_pat == -1:
            self.suggest_point["lr"] = round((self.left_lr+self.right_lr)/2, 1)
            self.suggest_point["start_lr"] = self.suggest_point["lr"]
            self.suggest_point["id"] = 0
            self.cur_pat = 1%self.patience_lr
            return copy.deepcopy(self.suggest_point), self.stopSignal
          elif self.cur_pat != 0:
            if self.added_model["data"]["accuracy"] > self.random_accuracy:
              self.suggest_point["id"] = self.cur_pat
              self.cur_pat = (self.cur_pat+1)%self.patience_lr
              return copy.deepcopy(self.suggest_point), self.stopSignal
            else:
              self.cur_pat = 0
          if self.cur_pat == 0 and round(self.right_lr-self.left_lr, 1) == 0.1:
            self._lrFound()
          elif self.cur_pat == 0:
            if self.added_model["data"]["accuracy"] > self.random_accuracy:
              self.left_lr = self.suggest_point["lr"]
            else:
              self.right_lr = self.suggest_point["lr"]
            if round(self.right_lr-self.left_lr, 1) == 0.1:
              self._lrFound()
            else:
              self.suggest_point["lr"] = round((self.left_lr+self.right_lr)/2, 1)
              self.suggest_point["start_lr"] = self.suggest_point["lr"]
              self.suggest_point["id"] = self.cur_pat
              self.cur_pat = (self.cur_pat+1)%self.patience_lr
              return copy.deepcopy(self.suggest_point), self.stopSignal


    if self.process_flag == "wd":
        if self.added_model["data"]["accuracy"] > self.best_model["data"]["accuracy"]:
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
            self.wd2 = 0
            self.wd2_deg = 0
            self.res2 = 0
            self.flag_wd = -1
          else:
            if self.res1 > self.res2:
              self.right_wd = self.wd2_deg
            if self.res2 > self.res1:
              self.left_wd = self.wd1_deg
            if self.res1 == self.res2:
              self.right_wd = self.wd2_deg
              self.left_wd = self.wd1_deg
        if self.flag_wd == 0:
          self._newWD()
          return copy.deepcopy(self.suggest_point), self.stopSignal
        if self.flag_wd == 1:
          self.res1 = self.added_model["data"]["accuracy"]
          self.flag_wd = 2
          self.suggest_point["wd"] = self.wd2
          self.suggest_point["flag"] = "EarlyStopping"
          return copy.deepcopy(self.suggest_point), self.stopSignal



    if self.process_flag == "random":
      overfit = False
      if self.best_model != {}:
        overfit, criterion = self._overfitting()
      if overfit == True:
        if criterion == "best":
          if self.best_model["data"]["accuracy"] >= self.added_model["data"]["accuracy"]:
            if self.prev_process != "wd":
              self.prev_process = "random"
              self._newWD()
              return copy.deepcopy(self.suggest_point), self.stopSignal
            else:
              overfit = False
          else:
            self.suggest_point = self.best_point
            self.best_point = self.added_point
            self.best_model = self.added_model
            self.prev_process = "random"
            self._newWD()
            return copy.deepcopy(self.suggest_point), self.stopSignal
        else:
          if self.best_model["data"]["accuracy"] >= self.added_model["data"]["accuracy"]:
            if self.prev_process != "wd":
              self.prev_process = "random"
              self._newWD()
              return copy.deepcopy(self.suggest_point), self.stopSignal
            else:
              overfit = False
          else:
            self.suggest_point = copy.deepcopy(self.added_point)
            self.best_point = self.added_point
            self.best_model = self.added_model
            self.prev_process = "random"
            self._newWD()
            return copy.deepcopy(self.suggest_point), self.stopSignal
      if overfit == False:
        if self.best_model != {}:
          if self.best_model["data"]["accuracy"] < self.added_model["data"]["accuracy"]:
            self.best_point = self.added_point
            self.best_model = self.added_model
        else:
            self.best_point = self.added_point
            self.best_model = self.added_model
        import random    
        parameter = random.choice(["test_loss", "test_accuracy"])
        mode = ""
        if parameter == "test_loss":
          mode = "min"
        else:
          mode = "max"
        if self.found_lr == -1:
          lr = round(random.uniform(0.1, 2), 5)
        else:
          lr = self.found_lr
        self.suggest_point = {
          "lr": lr,
          "wd": self.default_wd,
          "patience": random.randint(self.patience_left, self.patience_right),
          "warm_up": random.randint(self.warm_up_left, self.warm_up_right),
          "threshold": 10**-random.uniform(self.threshold_left, self.threshold_right), 
          "parameter": parameter,
          "mode": mode,
          "factor": 10**-random.uniform(self.factor_left, self.factor_right),
          "start_lr": self.warm_up_start,
          "momentum": 0.9,
          "flag": "EarlyStopping",
          "epochs": self.default_epochs,
          "id": 0
        }
        return copy.deepcopy(self.suggest_point), self.stopSignal 

  def why(self):
    if self.process_flag == "lr":
      return "Triyng to find the largest possible lr"
    if self.process_flag == "wd":
      return "Trying to reduce overfitting. Trying to find wd by ternary search. You can also use data augumentation"
    if self.process_flag == "random":
      return "Using The largest possible LR. Default wd = 0.0001. Other parameters are randomized"

  def stopSignal(self, data, flag, patience, parameter, epochs):
    import numpy as np
    patience += 5
    if flag == "default":
      if len(data["test_loss"]) == epochs:
        return True
      else:
        return False
    if flag == "EarlyStopping":
      if parameter == "test_accuracy":
        if len(data["test_accuracy"])-np.argmax(data["test_accuracy"]) > patience:
          return True
        else:
          return False
      elif parameter == "test_loss":
        if len(data["test_loss"])-np.argmin(data["test_loss"]) > patience:
          return True
        else:
          return False

  def _newWD(self):
    self.process_flag = "wd"
    self.wd1_deg = self.left_wd + (self.right_wd-self.left_wd)/3
    self.wd2_deg = self.right_wd - (self.right_wd-self.left_wd)/3
    self.flag_wd = 1
    self.wd1 = 10**-self.wd1_deg
    self.wd2 = 10**-self.wd2_deg
    self.suggest_point["wd"] = self.wd1
    self.suggest_point["flag"] = "EarlyStopping"
  def _lrFound(self):
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
    average_best_test = sum(self.best_model["data"]["test"+criterion]
                      [max(0, len(self.best_model["data"]["test"+criterion])-amount):
                      len(self.best_model["data"]["test"+criterion])-1])/amount
    average_best_train = sum(self.best_model["data"]["train"+criterion]
                       [max(0, len(self.best_model["data"]["train"+criterion])-amount):
                       len(self.best_model["data"]["train"+criterion])-1])/amount
    average_cur_test = sum(self.added_model["data"]["test"+criterion]
                     [max(0, len(self.added_model["data"]["test"+criterion])-amount):
                     len(self.added_model["data"]["test"+criterion])-1])/amount
    average_cur_train = sum(self.added_model["data"]["train"+criterion]
                      [max(0, len(self.added_model["data"]["train"+criterion])-amount):
                      len(self.added_model["data"]["train"+criterion])-1])/amount
    if average_best_test > average_cur_test and average_cur_train > average_best_train:
      return True, "cur"
    if average_best_test < average_cur_test and average_cur_train < average_best_train:
      return True, "best"
    criterion = "_loss"
    average_best_test = sum(self.best_model["data"]["test"+criterion]
                      [max(0, len(self.best_model["data"]["test"+criterion])-amount):
                      len(self.best_model["data"]["test"+criterion])-1])/amount
    average_best_train = sum(self.best_model["data"]["train"+criterion]
                       [max(0, len(self.best_model["data"]["train"+criterion])-amount):
                       len(self.best_model["data"]["train"+criterion])-1])/amount
    average_cur_test = sum(self.added_model["data"]["test"+criterion]
                     [max(0, len(self.added_model["data"]["test"+criterion])-amount):
                     len(self.added_model["data"]["test"+criterion])-1])/amount
    average_cur_train = sum(self.added_model["data"]["train"+criterion]
                      [max(0, len(self.added_model["data"]["train"+criterion])-amount):
                      len(self.added_model["data"]["train"+criterion])-1])/amount
    if average_best_test < average_cur_test and average_cur_train < average_best_train:
      return True, "cur"
    if average_best_test > average_cur_test and average_cur_train > average_best_train:
      return True, "best"
    return False, "none"
