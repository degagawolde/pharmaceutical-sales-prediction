import streamlit as st
import logging
import io
import os
import pandas as pd
from multipleapp import MultipleApp
# import your app modules here
from apps import data_information, user_overview, user_engagement, user_experience, user_satisfactin



app = MultipleApp()

# Add all your application her

app.add_app("Dataset Information", data_information.data_info, original_df)
app.add_app('User Overview Analysis', user_overview.user_overview, clean_data)
app.add_app('User Engegement Analysis',
            user_engagement.user_engagement, clean_data)
app.add_app('User Experience Analysis',
            user_experience.user_experience, original_df)
app.add_app('User Satisfaction Analysis',
            user_satisfactin.user_satisfaction, [])
# The main app
app.run()
