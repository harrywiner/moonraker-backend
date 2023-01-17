## MoonRaker ML API
Running

`python moonrakerbackend/manage.py runserver`


## Virtualenv
Setting up

```python -m venv venv```


Starting the environment, then create new terminal window

```.\venv\Scripts\activate.bat```

If you get an error saying 'cannot be loaded because running scripts is disabled on this system' then you go to powershell and type the command:

```RemoteSigned -Scope CurrentUser```

This changes the signing settings as outlined in [this document](https:/go.microsoft.com/fwlink/?LinkID=135170)

Install requirements

```pip install -r requirements.txt```

To sync the package, run

```pip freeze > requirements.txt```

