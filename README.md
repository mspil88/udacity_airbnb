# Boston Airbnb Analysis

This is the readme file for the analysis behind the blog post [How To Get Best Deals in Boston when Booking through Airbnb](https://medium.com/@matthewspillane88/how-to-get-best-deals-in-boston-when-booking-through-airbnb-e01da035cf51)

# Contents

* Installation
* Project description
* Files
* Outputs
* Licencing

## Installation

Minimum of Python version 3.6, remaining libraries will be contained within the Anaconda distribution list

## Project description

In this project I analyse 2017 Boston Airbnb listings data in order to understand the three considerations to be top of mind when making a booking in that City. 
In particular I focus on:

* What are the key drivers of prices in Boston and what implications do these drivers have when making a booking?

This opened up three key factors:

* Size of listing
* Location of listing
* The importance of considering potential hidden costs associated with your booked accomodation

## Files

The project has the following structure:

Root
|
|-----Data
|		|---- listings.csv
|		|---- calendar.csv
|		|---- reviews.csv (unused)
|		
|
|-----Utils
|		|---- analysis_utils.py
|		|---- data_proc.py
|
|-----business_questions.ipynb
|
|-----initial_eda.ipynb
|
|-----final_modelling.ipynb		

The data folder contains the data used for the project, the utils folder contains the customer modules I created, including for the purposes of processing the data and 
for analysing/ modelling. The business_questions.ipynb contains the analysis used for the blog post whilst the initial_eda.ipynb reflects the scratch work done at the beginning.
final_modelling.ipynb contains all of the modelling work which fed into the business_questions file.

## Outputs

The blog post can be found [here](https://medium.com/@matthewspillane88/how-to-get-best-deals-in-boston-when-booking-through-airbnb-e01da035cf51)

## Licencing

Credit goes to Airbnb and Udacity for the data. Anyone can use the code above if they see fit.

