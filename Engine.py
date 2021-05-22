import ISearch
class Engine():
  def __init__(self, _teach, _upload, _backup):
    self.seed = None
    self.process_flag = ""
    self.default_answer_flag = False
    self.teach = _teach
    self._upload = _upload
    self.backup = _backup
    self.data_buf = []
    self.point_buf = []
    self.best_model = dict()
    self.best_point = dict()
  def start(self, cur_point = None, stopSignal = None, method = None):
    if cur_point == None:
      method = ISearch()
      cur_point, stopSignal = method.nextPoint()
      print("proposed next point is: ", cur_point)
      if cur_point["flag"] == "default":
        print(cur_point["epochs"], " epochs")
      else:
        print(cur_point["flag"])
      print("\n\n\n")
      print("write 'change [lr=, wd=]' to change the next point")
      print("write 'do it' to set 'yes' as default answer for the process")
      print("to stop the algorithm write 'stop'")
      print("write 'yes' to use the suggested next point\n")
      ans = input().split(" ")
      while ans[0] != "yes":
        if len(ans) == 2 and ans[0] == "do" and ans[1] == "it":
          self.default_answer_flag = True
          self.process_flag = method.why()
          break
        elif ans[0] == "add":
          model, cur_point = self._addModel(cur_point, ans[1:])
          self.point_buf.append(cur_point)
          self.data_buf.append(model)
          break
        elif ans[0] == "change":
          cur_point = self._changePoint(ans[1:], cur_point)
          while True:
            print("write number of epochs or EarlyStopping")
            ans2 = input()
            if ans2.isdigit() == True:
              cur_point["epochs"] = int(ans2)
              cur_point["flag"] = "default"
              break
            if ans2 == "EarlyStopping":
              cur_point["flag"] = "EarlyStopping"
              break
          print("new point is ", cur_point)   
          if cur_point["flag"] == "default":
            print(cur_point["epochs"], " epochs")
          else:
            print(cur_point["flag"])
        elif ans[0] == "why":
          print(method.why())
        ans = input().split(" ")
      if ans[0] == "yes" or (len(ans) == 2 and ans[0] == "do" and ans[1] == "it"):
        self.point_buf.append(cur_point)
        self.data_buf.append(self.teach(cur_point, stopSignal, self.seed))
      method.addResult(self.data_buf[-1], self.point_buf[-1])
      self.best_point = self.point_buf[-1]
      self.best_model = self.data_buf[-1]
      print("\nnew bestModel\n")

    while True:
      if cur_point in self.point_buf:
        cur_point, stopSignal = method.nextPoint()
        self._backup(cur_point, stopSignal, method)
      if method.why() != self.process_flag:
        self.default_answer_flag = False
      if self.default_answer_flag == False:
        print("\n\n\n")
        print("proposed next point is: ", cur_point)    
        if cur_point["flag"] == "default":
          print(cur_point["epochs"], " epochs")
        else:
          print(cur_point["flag"])
        print("\n\n\n")
        print("write 'add' to add model")
        print("write 'change [lr=, wd=]' to change the next point")
        print("write 'do it' to set 'yes' as default answer for the process")
        print("write 'download [model indexes or best_model or cur_model]'" 
              " to download data and model")
        print("write 'info buf' to see hyperparameters buffer")
        print("to see accuracy write 'info accuracy [model indexes or best_model or cur_model]'")
        print("to see any plots write 'info [models model indexes or best_model or cur_model] " 
              "plots [tensorboard] [train_accuracy, test_accuracy, test_loss, train_loss, grad_min," 
              " grad_max, grad_average, grad_norm, weight_min, weight_max, weight_average, " 
              "weight_norm, lr_scheduler, momentum_cycle, lr_momentum]'")
        print("to stop the algorithm write 'stop'")
        print("write 'yes' to use the suggested next point\n")
      else:
        print("\n\n\n")
        print("next point is: ", cur_point)
        if cur_point["flag"] == "default":
          print(cur_point["epochs"], " epochs")
        else:
          print(cur_point["flag"])
        print("\n\n\n")
      while True:
        if self.default_answer_flag == True:
          self.point_buf.append(cur_point)
          self.data_buf.append(self.teach(cur_point, stopSignal, self.seed))
          method.addResult(self.data_buf[-1], self.point_buf[-1])
          if self.data_buf[-1]["data"]["accuracy"] > self.best_model["data"]["accuracy"]:
            print("\nnew bestModel\n")
            self.best_model = self.data_buf[-1]
            self.best_point = self.point_buf[-1]  
          break
        ans = input().split(" ")
        if len(ans) == 2 and ans[0] == "do" and ans[1] == "it":
            self.default_answer_flag = True
            self.process_flag = method.why()
        elif ans[0] == "add":
          model, cur_point = self._addModel(cur_point, ans[1:])
          if model != False:
            self.point_buf.append(cur_point)
            self.data_buf.append(model)
            method.addResult(self.data_buf[-1], self.point_buf[-1])
            if self.data_buf[-1]["data"]["accuracy"] > self.best_model["data"]["accuracy"]:
              print("\nnew bestModel\n")
              self.best_model = self.data_buf[-1]
              self.best_point = self.point_buf[-1]
            break
           
        elif ans[0] == "download":
          if (len(ans) == 1):
            self._download([self.data_buf[-1]], [str(self.point_buf[-1])])
          else:
            datas = []
            points = []
            for i in ans:
              if i[-1] == ',':
                i = i[:-1]
              if i.isdigit() and int(i) >= 0 and int(i) < len(self.point_buf):
                datas.append(self.data_buf[int(i)])
                points.append(str(self.point_buf[int(i)]))
              elif i == "best_model":
                datas.append(self.best_model)
                points.append(str(self.best_point) + "_bestModel")
              elif i == "cur_model":
                datas.append(self.data_buf[-1]["data"])
                points.append(str(self.point_buf[-1]) + "_cur_point")
            self._download(datas, points)
        elif ans[0] == "info":
          if ans[1] == "buf":
            self._buf()
          if ans[1] == "accuracy":
            self._accuracy(ans[2:])
          datas = []
          plots = []
          points = []
          flag = 0
          if ans[1] == "models":
            for i in ans:
              if i[-1] == ',':
                i = i[:-1]
              if i.isdigit() and int(i) >= 0 and int(i) < len(self.point_buf) and flag == 0:
                datas.append(self.data_buf[int(i)]["data"])
                points.append(str(self.point_buf[int(i)]))
              elif i == "best_model":
                datas.append(self.best_model["data"])
                points.append(str(self.best_point) + " bestModel")
              elif i == "cur_model":
                datas.append(self.data_buf[-1]["data"])
                points.append(str(self.point_buf[-1]) + " cur_point")
              if i == "plots":
                flag = 1
              if i == "tensorboard":
                flag = 2
              if i in self.data_buf[-1]["data"] and flag > 0:
                plots.append(i)
            if flag == 1:
              self._plot(datas, plots, points)
            if flag == 2:
              self._tensorboard(datas, plots, points)
          if ans[1] == "plots":
            for i in ans:
              if i[-1] == ',':
                i = i[:-1]
              if i == "tensorboard":
                flag = 2
              if i in self.data_buf[-1]["data"]:
                plots.append(i)
            if flag == 2:
              self._tensorboard([self.data_buf[-1]["data"]], plots, [str(self.point_buf[-1])])
            else:
              self._plot([self.data_buf[-1]["data"]], plots, [str(self.point_buf[-1])])
        elif ans[0] == "stop":
          return self.best_model, self.best_point
        elif ans[0] == "change":
          cur_point = self._changePoint(ans[1:], cur_point)
          while True:
            print("write number of epochs or EarlyStopping")
            ans2 = input()
            if ans2.isdigit() == True:
              cur_point["epochs"] = int(ans2)
              cur_point["flag"] = "default"
              break
            if ans2 == "EarlyStopping":
              cur_point["flag"] = "EarlyStopping"
              break
          print("new point is ", cur_point)   
          if cur_point["flag"] == "default":
            print(cur_point["epochs"], " epochs")
          else:
            print(cur_point["flag"])
        elif ans[0] == "yes":
          self.point_buf.append(cur_point)
          self.data_buf.append(self.teach(cur_point, stopSignal, self.seed))
          method.addResult(self.data_buf[-1], self.point_buf[-1])
          if self.data_buf[-1]["data"]["accuracy"] > self.best_model["data"]["accuracy"]:
            print("\nnew bestModel\n")
            self.best_model = self.data_buf[-1]
            self.best_point = self.point_buf[-1]
          break
        elif ans[0] == "why":
          print(method.why())
  
  def upload(self):
    my_files = self._upload()
    self.seed = my_files["seed"]
    self.default_answer_flag = my_files["default_answer_flag"]
    self.teach = my_files["teach"]
    self._upload = my_files["_upload"]
    self.backup = my_files["backup"]
    self.data_buf = my_files["data_buf"]
    self.point_buf = my_files["point_buf"]
    self.best_model = my_files["best_model"]
    self.best_point = my_files["best_point"]
    self.start(my_files["cur_point"], my_files["stopSignal"], my_files["method"])
    

  def _isDigit(self, x):
    try:
      float(x)
      return float(x) >= 0
    except ValueError:
      return False
  
  def _backup(self, cur_point, stopSignal, method):
    backup = {"seed": self.seed,
              "process_flag": self.process_flag,
              "default_answer_flag": self.default_answer_flag,
              "teach": self.teach,
              "_upload": self._upload,
              "backup": self.backup,
              "data_buf": self.data_buf,
              "point_buf": self.point_buf,
              "best_model": self.best_model,
              "best_point": self.best_point,
              "cur_point": cur_point,
              "stopSignal": stopSignal,
              "method": method,
    }
    self.backup(backup, len(self.point_buf))

  def _addModel(self, cur_point, ans):
    new_point = {}
    done = 0
    while True:
      for i in ans:
        splited = i.split("=")
        if (len(splited) == 2 and splited[0] != ""):
          if (splited[1][-1] == ','):
              splited[1] = splited[1][:-1]
          if (splited[0] in cur_point):
            if splited[0] == "parameter":
              if ("mode" not in new_point and splited[1] in ["test_loss", "test_accuracy"]) or (
                  "mode" in new_point and splited[1] == "test_loss" and new_point["mode"] == "min") or (
                  "mode" in new_point and splited[1] == "test_accuracy" and new_point["mode"] == "max"):
                new_point.update({splited[0]: splited[1]})
              else:
                print("wrong parameter")
            elif splited[0] == "mode":
              if ("parameter" not in new_point and splited[1] in ["max", "min"]) or (
                  "parameter" in new_point and splited[1] == "min" and new_point["parameter"] == "test_loss") or (
                  "parameter" in new_point and splited[1] == "max" and new_point["parameter"] == "test_accuracy"):
                new_point.update({splited[0]: splited[1]})
              else:
                print("wrong mode")
            elif splited[0] == "flag":
              if splited[1] == "EarlyStopping" or splited[1] == "default":
                new_point.update({splited[0]: splited[1]})
              else:
                print("wrong flag")
            elif self._isDigit(splited[1]):
              new_point.update({splited[0]: float(splited[1])})
      if len(new_point) == len(cur_point):
        break
      print("not enough parameters")
      print("try more or write break to stop adding model")
      ans = input()
      if ans == "break":
        return False, False
      ans = ans.split(" ")
    from google.colab import files
    import pickle
    import io
    new_data = {}
    while True:
      my_files = files.upload()
      if len(my_files) != 2:
        print("please, choose only two files")
      else:
        flag = 0
        for j in my_files:
          if ".pkl" in j:
            data = pickle.load(io.BytesIO(my_files[j]))
            for i in data:
              if "error" in i:
                new_data.update({i.replace("error", "accuracy"): data[i]})
              else:
                new_data.update({i: data[i]})
            flag += 1
          if ".pth" in j:
            model = my_files[j]
            flag+=1
        if flag == 2:
          break
        else:
          print("please, choose two files: .pkl, .pth")

    import copy
    return copy.deepcopy({"data": new_data, "model": model}), new_point

  def _plot(self, datas, plots, points):
    import matplotlib.pyplot as plt
    import numpy as np
    for i in plots:
      %matplotlib inline
      fig = plt.figure()
      axes = fig.add_subplot(111)
      for j in range(len(datas)):
        if np.array(datas[j][i]).ndim == 1:
          axes.plot(datas[j]["x"], datas[j][i], label=points[j])
        else:
          out = [datas[j][i][k][0] for k in range(len(datas[j][i]))]
          axes.plot(datas[j]["x"], out, label=points[j])
      handles, labels = axes.get_legend_handles_labels()
      lgd = plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,-0.2))
      axes.set_title(i)
      axes.grid('on')
      axes.set_xlabel('iteration')
      axes.set_ylabel(i)
      plt.show()

  def _tensorboard(self, datas, plots, points):
    import numpy as np
    from torch.utils.tensorboard import SummaryWriter
    %load_ext tensorboard
    for i in range(len(datas)):
      writer = SummaryWriter(comment ="\\ " + points[i] + " \\")
      for j in plots:
        if np.array(datas[i][j]).ndim == 1:
          for k in range(len(datas[i]["x"])):
            writer.add_scalar(j, datas[i][j][k], datas[i]["x"][k])
        else:
          for k in range(len(datas[i]["x"])):
            writer.add_scalar(j, datas[i][j][k][0], datas[i]["x"][k])
      writer.close()
    %tensorboard --logdir runs

  def _accuracy(self, ans):
    if (len(ans) == 0):
            print(self.point_buf[-1], " ", self.data_buf[-1]["data"]["accuracy"])
    else:
      for i in ans:
        if i[-1] == ',':
          i = i[:-1]
        if i.isdigit() and int(i) >= 0 and int(i) < len(self.point_buf):
          print(self.point_buf[int(i)], " ", self.data_buf[int(i)]["data"]["accuracy"])
        elif i == "best_model":
          print(self.best_point, " bestModel ", self.best_model["data"]["accuracy"])
        elif i == "cur_model":
          print(self.point_buf[-1], " cur_point ", self.data_buf[-1]["data"]["accuracy"])

  def _download(self, datas, points):
    for i in range(len(points)):
      torch.save(datas[i]["model"], "model_" + points[i] + ".pth")
      files.download("model_" + points[i] + ".pth")
      a_file = open("data_" + points[i] + ".pkl", "wb")
      pickle.dump(datas[i]["data"], a_file)
      files.download("data_" + points[i] + ".pkl")

  def _buf(self):
    for i in range(len(self.point_buf)):
      print(i, " ", self.point_buf[i])

  def _changePoint(self, ans, point):
    import copy
    for i in ans:
      splited = i.split("=")
      if (len(splited) == 2 and splited[0] != ""):
        if (splited[1][-1] == ','):
            splited[1] = splited[1][:-1]
        if (splited[0] in point):
            if splited[0] == "parameter":
              if splited[1] == "test_loss":
                point["mode"] = "min"
                point[splited[0]] = splited[1]
              elif splited[1] == "test_accuracy":
                point["mode"] = "max"
                point[splited[0]] = splited[1]
              else:
                print("wrong parameter")
            elif splited[0] == "mode":
              if splited[1] == "min":
                point["parameter"] = "test_loss"
                point[splited[0]] = splited[1]
              elif splited[1] == "max":
                point["parameter"] = "test_accuracy"
                point[splited[0]] = splited[1]
              else:
                print("wrong mode")
            elif splited[0] == "flag":
              if splited[1] == "EarlyStopping" or splited[1] == "default":
                point[splited[0]] = splited[1]
              else:
                print("wrong flag")
            elif self._isDigit(splited[1]):
                point[splited[0]] = float(splited[1])
    return copy.deepcopy(point)
