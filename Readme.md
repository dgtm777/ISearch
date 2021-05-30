Программа написана для использования в colab.research.google.com

Программа состоит из двух классов. Первый отвечает за связь пользователея с алгоритмом автоматического подбора. Будем называть его Engine. 
Второй отвечает за подбор параметров. Назовем его ISearch по названию метода. Основными методоми класса ISearch являются функции next_point, add_result и why. 
next_point возвращает предлагаемую алгоритмом следующую точку дла обучения. Чтобы определить следующую точку ISearch сохраняет у себя наилучшую обученную модель. 
Сделать это позволяет функция add_result. Класс ISearch не влияет на сам процесс обучения. Он предлагает гиперпараметры для обучения и потом получает результаты. 
Функция why позволяет пользователю запросить причины выбора следующей точки обучения. 

Engine имеет два доступных пользователю метода - start и upload. start запускает метод с начала, upload позволяет запустить метод с последнего сохранения.
Стоит отметить, что в функцию add_result могут прийти статистики обучения не той точки, которою предлагал метод, а той, которую выбрал пользователь. 
Пользователь в свою очередь должен передать в конструктор Engine функцию, которая по заданным гиперпараметрам обучит свою сетку и вернет статистики обучения и саму модель. 

Функция пользователя помимо гиперпараметров для обучения должна принимать на вход функцию из класса ISearch, которая определит, когда обучение нужно прекратить работу. Работа прекращается либо когда истекло заданное колличество эпох, либо, когда сетка начала переобучаться. Так реализуется ранняя остановка (early stopping) в пользовательской функции. Таким образом, всего есть три блока: функция, предоставляемая пользователем принимает на вход точку и возвращает статистики, класс ISearch, выбирающий гиперпараметры для обучения, и Engine, который осуществляет связь между пользователем и методом.


https://colab.research.google.com/drive/1aqx-5DzIs4TWW0WmDo-vsVWv49XeG5gU?usp=sharing
