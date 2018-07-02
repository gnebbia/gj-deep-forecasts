# TODO List
* Insert distance from outcome directly and normalize it between [-1,1]
* Does loss function need an average?
* work on the jupyter notebook for the training of the model
* make the LDA model parameterized with number of topics and choice of
  text/description
* add a description for each file/directory of the project
*? Implement user Ids as category with a:
```python
  df_complete["user_id_1_code"] = df_complete["user_id_1"].cat.codes
  df_complete["user_id_2_code"] = df_complete["user_id_2"].cat.codes
```

?: means "do we need this?"
