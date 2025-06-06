# VolatilePoolSimulations

1. The ContractVolatile.py includes functions details needed for the volatile pool design and ensembles the smart contract functions and we tune all parameters from that file.
2. MarketArb.py sets up the environment to imitate a DEX environment. It assumes the basic DEX environment with initial depositor and Arbitraguer and regular LP. The simulation is conducted through a historic data and export a json file for visualization and parameter tuning.
3. Unitest.ipynb provides simple script that allows users to set the state and test different functions.
4. json_graphing.py visualizes the analysis needed in the yellowpaper.
5. BTC_ETH_price.csv is the historic data covering the period from 1st May, 2021 to 2022.
6. data_export.json is the simulated result.

It achieve several goals:
1. Testing System Vulnerability 
2. According to historical data (hourly) → How different parameters change the fee and repegging behaviours. 
3. Parameter Tuning Suggestions
4. Observing functions output through pre-setting certain state.
5. Estimating the Impermanent loss for regular LP under specific scenario e.g. BTC drop by x %.

Wombat Community can simulate the situation under different settings including number of tokens, change in price, swapping volume...etc. 
As such, it provides an empirical guideline for the governance and balance our LP benefits through any parameters settings if needed. 
