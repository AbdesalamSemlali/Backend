from datetime import datetime, date

def get_date_difference(date_string):
    date_object = datetime.strptime(date_string,"%d-%m-%Y").date()
    today = date.today()
    date_difference =  date_object - today
    return date_difference.days