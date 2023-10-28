# minio-with-pl
## Usage
1. Поднимаем контейнер: `docker-compose up` (или `up -d` для detached режима)
2. Пробрасываем порты `9090` и `9000`
3. Логинимся в http://localhost:9090/
4. Заводим бакет, конфиругируем свойства
5. Создаём Access_Key. Нам покажут Secret_key единожды. После этого необходимо придумать себе какое-нибудь профиль-имя и создать файл `~/.aws/credentials` со сделующим содержимым:
```bash
[<profile>]
aws_access_key_id = <access_key>
aws_secret_access_key = <secret_key>
```
6. Меняем конфиг в `conf/config.yaml` с указанием соответствующего профиля и бакета, настраиваем остальные параметры по вкусу
7. Устанавливаем зависимости
8. Стартуем обучение `./train.py`, прерываем в середине
9. Через браузер убеждаемся, что чекпоинты в бакете присутствуют
10. Восстанавливаем обучение с последнего чекноинта, предварительно выставив флаг `use_ckpt` в `True` и указав нужный нам чекпоинт
11. Доводим обучение до конца, меняем `ckpt_path` на полностью обученную модель, тестируем `./test.py`

Для работы через различных клиентов
- `MINIO_ROOT_USER` = `user_name`
- `MINIO_ROOT_PASSWORD` = `your_password`

  ## References
  
- [Lightning MNIST tutorial](https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/mnist-hello-world.html) 
- [Using checkpoint Plugin in Lightning](https://lightning.ai/docs/pytorch/stable/common/checkpointing_expert.html#checkpointing-expert) 
