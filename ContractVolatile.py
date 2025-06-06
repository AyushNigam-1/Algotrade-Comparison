"""
Created on Tue 5 Dec, 2023. 14:35

@author: Andy Lee
"""
import numpy as np
import pandas as pd
import gc

# System infinitly small positive number, for testing extreme situation
epsilon = 1e-30

# Constants
print_update = False
print_action = False
no_fee = False
Force_withdraw = False
deposit_withdraw_repeg = False

######### Configurable parameters
# Amm factor, to be modified
A = 0.99

# Dynamic fee parameters
t = 1  # Modified by ema_t.
T = 24  # The oracle hours

# Dynamic fee parameters
beta_v = 0.95  # for s.d around 0.5
k_v = 12
beta_v2 = 1.33  # for s.d around 1.2
k_v2 = 45
beta_U = 25  # Utilization rate between 2-5%
k_U = 0.3  #  Utilization rate between 2-5%
# Imbalance fee parameters
theta = 5000000000000  # With BTC drop by 20%, coverage ratio 0.8-1.2
theta_exp = 32  # With BTC drop by 20%, coverage ratio 0.8-1.2, when r=0.8, fee=fee_imbalance *1.5
# Fee Base parameters
Fee_base = 0.000325
Fee_sigma_u = 0.002
Fee_imbalance = 0.000125

# Force withdrawal parameters, at this parameter, can't reduce coverage ratio after 0.7r, improve if smaller than 0.7
A_omega = 40000000
k_omega = 25

# LP Share of swap fee
rrlp = 1  # amount of fee that is going to the LP, The rest is going to the protocol
repeg_ratio = 0.5  # portion of LP fee used for repegging.


# Repegging parameters
step = 0.0009  # 0.00049 # curve 49*10**13 where price_ratio=p_oracle/p_scale *10**18

operation_counters = {
    "swap_quote": 0,
    "calculate_fee": 0,
    "tweak_price": 0,
    "repeg": 0
}


# Custom Functions
def add_or_assign(name, dictionary, amount):
    if name in dictionary.keys():
        dictionary[name] += amount
    else:
        dictionary[name] = amount


def equilCovRatio(from_token, to_token, other_token):  # r* calculation
    tokens = [from_token, to_token, other_token]
    D = 0
    sum_L = 0
    for token in tokens:
        A_i = token.cash
        L_i = token.liability
        P_i = token.price_scale
        if L_i == 0:
            continue
        r_i = A_i / L_i
        sum_L += P_i * L_i
        D += P_i * L_i * (r_i - A / r_i)
    if sum_L == 0:
        r_star = 1
        return r_star
    b = -(D / sum_L)
    r_star = (-b + np.sqrt(b * b + 4 * A)) / 2
    return r_star


def equilCovRatio_repeg(
    from_token, to_token, other_token
):  # r* after repegging with the new price and fee.
    tokens = [from_token, to_token, other_token]
    D = 0
    sum_L = 0

    for token in tokens:
        A_i = token.cash + token.hf_repeg
        L_i = token.liability
        P_i = token.price_scale_new
        if L_i == 0:
            continue
        r_i = A_i / L_i
        sum_L += P_i * L_i
        D += P_i * L_i * (r_i - A / r_i)
    if sum_L == 0:
        r_star_repeg = 1
        return r_star_repeg
    b = -(D / sum_L)
    r_star_repeg = (-b + np.sqrt(b * b + 4 * A)) / 2
    return r_star_repeg


def price_factor(from_token, to_token):  # For AMM
    relative_price = from_token.price_scale / to_token.price_scale
    return relative_price


def mintfee(
    token, amount
):  # swap fee return to LP as a form of deposit (to get deposit gain)
    reward_amount = 0
    delta_a = amount  # absolute amount of token deposit without gain
    A_i = token.asset()
    L_i = token.liab()

    if L_i == 0:  # L_i==0 is = First deposit
        reward_amount = amount
    elif A_i + delta_a < 0:
        print("!DEPOSIT ERROR")
        pass
    else:  # Calculate deposit gain
        r_i = A_i / (L_i)
        if r_i == 0:  # for extreme case
            r_i = epsilon
        k = A_i + delta_a
        b = k * (1 - A) + 2 * A * L_i
        if r_i != 1:
            c = k * (A_i - (A * L_i) / r_i) - k * k + A * (L_i) * (L_i)
            l = b * b - 4 * A * c
            reward_amount = (-b + np.sqrt(l)) / (2 * A)
        else:
            reward_amount = amount

    if no_fee == True:
        reward_amount = amount

    token.cash += amount

    # Liability to be added in the swap process
    return reward_amount


def swap_quote(  # For volatile pool swapping
    from_asset, to_asset, from_liab, to_liab, from_amount, A
):  # For quoting the swap amount.
    # Managing extreme cases; i.e. x=episolon
    operation_counters["swap_quote"] += 1
    ft_liab = epsilon if from_liab == 0 else from_liab
    ft_asset = epsilon if from_asset == 0 else from_asset

    tt_liab = epsilon if to_liab == 0 else to_liab
    tt_asset = epsilon if to_asset == 0 else to_asset

    # This is different from the normal swap because to and from token are not specified, just aim to return the amount
    ft_cov_ratio = epsilon if from_liab == 0 else from_asset / from_liab
    tt_cov_ratio = epsilon if to_liab == 0 else to_asset / to_liab

    if ft_cov_ratio == 0:
        ft_cov_ratio = epsilon
    if tt_cov_ratio == 0:
        tt_cov_ratio = epsilon

    D_swap = (ft_liab * ft_cov_ratio + tt_liab * tt_cov_ratio) - A * (
        (ft_liab / ft_cov_ratio) + (tt_liab / tt_cov_ratio)
    )
    temp_cov = (ft_asset + from_amount) / ft_liab

    b_swap = (ft_liab / tt_liab) * (temp_cov - A / temp_cov) - (D_swap / tt_liab)

    r_to_swap = (-b_swap + np.sqrt(b_swap**2 + 4 * A)) / 2

    to_amount = abs((r_to_swap * tt_liab - tt_asset))

    if print_update == True:
        print(
            f"ft_liab, ft_asset, tt_liab, tt_asset, ft_cov_ratio, tt_cov_ratio are {ft_liab},{ft_asset},{tt_liab},{tt_asset},{ft_cov_ratio},{tt_cov_ratio}"
        )
        print(
            "SWAP UPDATE: D_swap, temp_cov,b_swap, r_to_swap and to_amount are {0}, {1}, {2}, {3},{4}".format(
                D_swap, temp_cov, b_swap, r_to_swap, to_amount
            )
        )

    return to_amount


def swap_price(from_token, to_token):  # for getting price_last
    p_x = from_token.price_scale
    p_y = to_token.price_scale

    price = (p_x + (p_x * A / (from_token.cov_ratio()) ** 2)) / (
        (p_y + p_y * A / to_token.cov_ratio() ** 2)
    )
    return price


def tweak_price(
    from_token, to_token, other_token
):  # For updating price after each swap/Deposit/withdraw, also trigger repegging
    operation_counters["tweak_price"] += 1
    tokens = [from_token, to_token, other_token]
    stablecoin = None
    alpha = 0
    for token in tokens:
        if token.name == "USDT":
            stablecoin = token
            ema_time = 3000000
            alpha = 2 ** (
                -stablecoin.ema_t  # Note that the average of maximum swapping ema_t per hour is around 600.
                / ema_time  # This is resemble the timing component of the block.time which increasing the impact of oracle through increasing duration, we done it through the number of swap within an hour.
            )

    for token in tokens:
        if token.name != "USDT":  # Only update the price of non-numeraire
            token.price_last = swap_price(token, stablecoin)
            if token.price_last > token.price_scale:
                token.price_oracle = (
                    np.minimum(token.price_last, 2 * token.price_scale) * (1 - alpha)
                    + token.price_oracle * alpha
                )
            else:
                token.price_oracle = (
                    np.maximum(token.price_last, token.price_scale / 2) * (1 - alpha)
                    + token.price_oracle * alpha
                )

    # For repegging and new price_scale
    sum_norm = 0
    norm = 0
    for token in tokens:
        ratio = token.price_oracle / token.price_scale
        if ratio > 1:
            ratio = ratio - 1
        else:
            ratio = 1 - ratio
        sum_norm += (ratio) ** 2
        norm = np.sqrt(sum_norm)
    adjustment_step = np.maximum(step, norm / 5)
    if norm > adjustment_step:
        for token in tokens:
            if token.name != "USDT":
                token.price_scale_new = (
                    token.price_scale * (norm - adjustment_step)
                    + adjustment_step * token.price_oracle
                ) / norm

    # New repegging based on r*
    r_star_repeg = equilCovRatio_repeg(from_token, to_token, other_token)
    if r_star_repeg > 1:
        print(
            "******************************* Repegging Triggered *******************************"
        )
        operation_counters["repeg"] += 1
        for token in tokens:
            if token.price_scale_new != token.price_scale:
                print(
                    f"Repegging {token.name} from {token.price_scale} to {token.price_scale_new} (oracle: {token.price_oracle})"
                )
                token.cash += token.hf_repeg
                token.hf_repeg -= token.hf_repeg
                token.price_scale = token.price_scale_new


def tokens_records(from_token, to_token, other_token):  # For analysis use
    tokens = [from_token, to_token, other_token]
    for token in tokens:
        token.price_last_records.append(token.price_last)
        token.price_oracle_records.append(token.price_oracle)
        token.price_scale_records.append(token.price_scale)
        token.price_scale_new_records.append(token.price_scale_new)
        token.hf_repeg_records.append(token.hf_repeg)
        token.cov_ratio_records.append(token.cov_ratio())
        token.cash_records.append(token.cash)


def sigma_T(from_token, to_token):  # Dynamic fee sigma amended
    ft_average_prices_records = [
        recorded_price
        for recorded_price in from_token.price_last_average_records
        if recorded_price != 1e10  # Extracting hourly average price
    ]

    tt_average_prices_records = [
        recorded_price
        for recorded_price in to_token.price_last_average_records
        if recorded_price != 1e10  # Extracting hourly average price
    ]

    # Dealing with first few hours
    if np.minimum(len(ft_average_prices_records), len(tt_average_prices_records)) < T:
        T_window = np.minimum(
            len(ft_average_prices_records), len(tt_average_prices_records)
        )
    else:
        T_window = T

    # calculate the average price
    ft_average_prices = np.array(ft_average_prices_records[-T_window:])

    tt_average_prices = np.array(tt_average_prices_records[-T_window:])

    # Calculate the percentage change  of average price
    if T_window < 2:  # dealing with the first item
        ft_sigma_T = 0
        tt_sigma_T = 0
    else:
        ft_return = [
            np.log(ft_average_prices[i] / ft_average_prices[i - 1]) * 100
            if i > 0
            else 0
            for i in range(1, len(ft_average_prices))
        ]

        tt_return = [
            np.log(tt_average_prices[i] / tt_average_prices[i - 1]) * 100
            if i > 0
            else 0
            for i in range(1, len(tt_average_prices))
        ]

        ft_mean = np.mean(ft_return)
        tt_mean = np.mean(tt_return)

        ft_ssd = np.sum([(price_change - ft_mean) ** 2 for price_change in ft_return])
        tt_ssd = np.sum([(price_change - tt_mean) ** 2 for price_change in tt_return])

        ft_sigma_T = np.sqrt(ft_ssd / len(ft_return))
        tt_sigma_T = np.sqrt(tt_ssd / len(tt_return))

    if from_token.name != "USDT" and to_token.name != "USDT":
        sigma_T = (ft_sigma_T + tt_sigma_T) / 2
    elif from_token.name == "USDT":
        sigma_T = tt_sigma_T
    elif to_token.name == "USDT":
        sigma_T = ft_sigma_T

    to_token.sigma_T_records.append(tt_sigma_T)
    from_token.sigma_T_records.append(ft_sigma_T)
    return sigma_T


def utilization_T(from_token, to_token):  # not used for the latest design
    # Calculate Liquidity
    L = np.sqrt(
        to_token.cash * to_token.price_scale * from_token.cash * from_token.price_scale
    )

    # Extracting hourly average values
    tt_volume_records = [
        recorded_volume
        for recorded_volume in to_token.volume_hourly_records
        if recorded_volume != 1e10
    ]

    ft_volume_records = [
        recorded_volume
        for recorded_volume in from_token.volume_hourly_records
        if recorded_volume != 1e10
    ]

    # Dealing with first few hours
    if len(tt_volume_records) < 1:
        tt_volume_records = [0]
        ft_volume_records = [0]

    if np.minimum(len(tt_volume_records), len(ft_volume_records)) < 1:
        tt_volume_records = [0]
        ft_volume_records = [0]

    if np.minimum(len(tt_volume_records), len(ft_volume_records)) < T:
        T_window = np.minimum(len(tt_volume_records), len(ft_volume_records))
    else:
        T_window = T

    # Calculate volume
    tt_total_volume = np.sum(tt_volume_records[-T_window:])
    ft_total_volume = np.sum(ft_volume_records[-T_window:])
    V = np.sqrt(tt_total_volume * ft_total_volume)

    # Calculate Trade to liquidity ratio
    utilization_T = V / L * 100
    to_token.utilization_T_records.append(utilization_T)
    return utilization_T


def calculate_fee(from_token, to_token):
    operation_counters["calculate_fee"] += 1
    # Calculate the volatility weight
    sigma = sigma_T(from_token, to_token)
    volatility_weight = 1 / (1 + np.exp(k_v * (beta_v - sigma)))
    volatility_weight2 = 1 / (1 + np.exp(k_v2 * (beta_v2 - sigma)))

    # Calculate the utilization weight
    utilization = utilization_T(from_token, to_token)

    utilization_weight = 1 / (1 + np.exp(k_U * (beta_U - utilization)))
    utilization = False
    if utilization == False:
        utilization_weight = 1
    # Calculate the imbalance level weight
    imbalance_weight = (
        theta * np.exp(-theta_exp * from_token.temp_cov_ratio())
        + theta * np.exp(-theta_exp * to_token.temp_cov_ratio())
    ) / 2
    # Calculate the overall fee
    dynamic_fee = (
        Fee_base
        + Fee_sigma_u * (volatility_weight + volatility_weight2) * utilization_weight
        + Fee_imbalance * (imbalance_weight)
    )

    # Records for analysis
    fee_u_record = Fee_sigma_u * volatility_weight * utilization_weight
    fee_imbalance_record = Fee_imbalance * (1 + imbalance_weight)
    to_token.imbalance_records.append(imbalance_weight)
    to_token.Fee_sigma_u_records.append(fee_u_record)
    to_token.Fee_imbalance_records.append(fee_imbalance_record)
    return dynamic_fee


class Users:
    def __init__(self, name):
        self.name = name
        self.wallet = {}
        self.lp_token = {}

    def reset(self):
        self.wallet = {"BTC": 10000, "ETH": 150000, "USDT": 300000000}
        self.lp_token = {"BTC": 0, "ETH": 0, "USDT": 0}

    @classmethod
    def get_instances(cls):
        result = []
        for obj in gc.get_objects():
            if isinstance(obj, cls):
                result.append(obj)
        return result

    def deposit(self, pool, token, amount, other_token1, other_token2):
        pool.deposit(self, token, amount, other_token1, other_token2)

    def withdraw_from_another_asset(
        self, pool, token, amount, other_token1, other_token2
    ):
        pool.withdraw_from_another_asset(
            self, token, amount, other_token1, other_token2
        )

    def withdraw_volatile(self, pool, token, amount, other_token1, other_token2):
        pool.withdraw_volatile(self, token, amount, other_token1, other_token2)

    def swap_volatile_repeg(self, pool, from_token, from_amount, to_token, other_token):
        pool.swap_volatile_repeg(self, from_token, from_amount, to_token, other_token)


class Tokens:
    def reset(self):  # for external auditing
        self.cash = 0
        self.lp_token = 0
        self.liability = 0
        self.price_scale = 1
        self.price_scale_new = 1
        self.price_oracle = 1
        self.price_market = 1
        self.price_last = 1
        self.temp_withdraw = 0
        self.temp_swap = 0
        self.hf_repeg = 0
        self.volume_records = [0]
        self.price_scale_records = [1]
        self.price_scale_new_records = [1]
        self.price_oracle_records = [1]
        self.price_last_records = [1]
        self.cov_ratio_records = [1]
        self.h_rate_records = [0]
        self.hf_repeg_records = [0]
        self.hf_lp_records = 0
        self.hf_sys_records = 0
        self.cash_records = [0]
        self.sigma_T_records = [0]
        self.utilization_T_records = [0]
        self.imbalance_records = [0]
        self.Fee_imbalance_records = [0]
        self.Fee_sigma_u_records = [0]
        self.price_last_dict_records = {}
        self.price_last_average_records = []
        self.volume_dict_records = {}
        self.volume_hourly_records = [0]
        self.volume = 0
        self.price_last_average = 1
        self.price_oracle_average = 1
        self.ema_t = 1

    def __init__(self, name):
        self.name = name
        self.cash = 0
        self.lp_token = 0
        self.liability = 0
        self.price_scale = 1
        self.price_scale_new = 1
        self.price_oracle = 1
        self.price_market = 1
        self.price_last = 1
        self.temp_withdraw = 0
        self.temp_swap = 0
        self.hf_repeg = 0
        self.volume_records = [0]
        self.price_scale_records = [1]
        self.price_scale_new_records = [1]
        self.price_oracle_records = [1]
        self.price_last_records = [1]
        self.cov_ratio_records = [1]
        self.h_rate_records = [0]
        self.hf_repeg_records = [0]
        self.hf_lp_records = 0
        self.hf_sys_records = 0
        self.cash_records = [0]
        self.sigma_T_records = [0]
        self.utilization_T_records = [0]
        self.imbalance_records = [0]
        self.Fee_imbalance_records = [0]
        self.Fee_sigma_u_records = [0]
        self.price_last_dict_records = {}
        self.price_last_average_records = []
        self.volume_dict_records = {}
        self.volume_hourly_records = [0]
        self.volume = 0
        self.price_last_average = 1  # New
        self.price_oracle_average = 1  # New
        self.ema_t = 1

    @classmethod
    def get_instances(cls):
        result = []
        for obj in gc.get_objects():
            if isinstance(obj, cls):
                result.append(obj)
        return result

    def asset(self):
        return self.cash

    def liab(self):
        return self.liability

    def lp_supply(self):
        return self.lp_token

    def price(self):
        return self.price_scale

    def cov_ratio(self):
        return 0 if self.liab() == 0 else self.asset() / self.liab()

    def temp_asset(self):
        return self.cash + self.temp_swap - self.temp_withdraw

    def temp_liab(self):
        return self.liability - self.temp_withdraw

    def temp_cov_ratio(self):
        return 0 if self.temp_liab() == 0 else self.temp_asset() / self.temp_liab()


class Pool:
    def __init__(self):
        return

    def deposit(self, user, token, amount, other_token1, other_token2):
        if amount > user.wallet[token.name]:
            amount = user.wallet[token.name]
        if amount <= 0:
            if print_update == True:
                print(
                    "!UNABLE TO DEPOSIT {1}: {0} has EMPTY balance ".format(
                        user.name, token.name
                    )
                )
            return

        reward_amount = 0
        delta_a = amount  # absolute amount of token deposit without gain
        A_i = token.asset()
        L_i = token.liab()
        if L_i == 0:
            reward_amount = amount
        elif A_i + delta_a < 0:
            print(f"A_i is {A_i}")
            print(f"delta_a is {delta_a}")
            print("!!!DEPOSIT ERROR")
            pass
        else:
            r_i = A_i / (L_i)
            if r_i == 0:  # For extreme case
                r_i = epsilon
            k = A_i + delta_a  # delta_a is D_i compared to the smart contract
            b = k * (1 - A) + 2 * A * L_i

            if r_i != 1:
                c = k * (A_i - (A * L_i) / r_i) - k * k + A * (L_i) * (L_i)
                l = b * b - 4 * A * c
                reward_amount = (-b + np.sqrt(l)) / (2 * A)
            else:
                reward_amount = amount

        if no_fee == True:
            reward_amount = amount

        # Adding token asset and subtracting user asset
        add_or_assign(token.name, user.wallet, -amount)
        token.cash += amount
        # Allocating lp token.
        lpTokenToMint = (
            reward_amount if L_i == 0 else reward_amount * token.lp_supply() / L_i
        )
        add_or_assign(token.name, user.lp_token, lpTokenToMint)
        token.lp_token += lpTokenToMint
        token.liability += reward_amount

        reward = reward_amount - amount
        reward_perc = reward / amount * 100

        if (
            deposit_withdraw_repeg == True
        ):  # deposit and withdraw also trigger repegging
            if (
                token.liability == 0
                or other_token1.liability == 0
                or other_token2.liability == 0
            ):
                return
            else:
                tweak_price(token, other_token1, other_token2)

        if print_action == True:
            print(
                "DEPOSIT: {0} deposited {1} {2} and gain reward of {3} ({4}%) adding total {5} {2} liability ({6} LP) into the Pool".format(
                    user.name,
                    amount,
                    token.name,
                    reward,
                    reward_perc,
                    reward_amount,
                    lpTokenToMint,
                )
            )

    def withdraw_from_another_asset(self, user, token, amount, to_token, other_token):
        withdraw_amount = self.withdraw_volatile(
            user, token, amount, to_token, other_token
        )
        self.swap_volatile_repeg(user, token, withdraw_amount, to_token, other_token)

    def withdraw_volatile(self, user, token, amount, other_token1, other_token2):
        if (
            user.lp_token[token.name] == 0 or token.lp_supply() == 0
        ):  # If LP wallet is empty
            if print_action == True:
                print(
                    "!UNABLE TO WITHDRAW {1}: {0} has EMPTY lp_balance ".format(
                        user.name, token.name
                    )
                )
            return

        A_i = token.asset()
        L_i = token.liab()
        if (
            amount > user.lp_token[token.name]
        ):  # If we withdraw more than the wallet amount
            delta_lp = -user.lp_token[token.name]
        else:
            delta_lp = (user.lp_token[token.name] / user.lp_token[token.name]) * (
                -amount  # else it's just the original amount
            )
        # Core Calculation
        liabilityToBurn = delta_lp * (L_i / token.lp_supply())  # This is <0
        L_i_ = L_i + liabilityToBurn
        r_i = A_i / (L_i)
        rho = L_i * (r_i - (A / r_i))
        beta = (rho + liabilityToBurn * (1 - A)) / 2
        A_i_ = beta + np.sqrt(beta**2 + (A * (L_i_**2)))
        withdraw_amount = A_i - A_i_

        # Skip: For taking away withdrawal fee
        if no_fee == True:
            withdraw_amount = abs(liabilityToBurn)
        # Skip: Only happens if we waive fee, and drained the pool
        if withdraw_amount > token.cash:
            withdraw_amount = token.cash
            liabilityToBurn = token.cash
            delta_lp = -token.cash * (token.lp_supply() / L_i)  # should be negative.

        # For understanding purposes
        if print_update == True:
            print(
                "WITHDRAW: {0} chooses {1} {4} LP token (choosing amount = {2}), liability to burn is  {3} {4}".format(
                    user.name, delta_lp, amount, liabilityToBurn, token.name
                )
            )

        #### Withdraw Stability function
        force_swap = 0
        to_token = None
        other = None
        if Force_withdraw == True:
            # Force withdraw in another token
            omega = np.maximum(
                A_omega * np.exp(-k_omega * r_i), 1
            )  # cap the omega =1 #and will only start

            force_swap = omega * (L_i_) * (r_i - ((A_i - withdraw_amount) / L_i_))
            if force_swap > withdraw_amount:
                force_swap = withdraw_amount

            # Determine which token to swap to
            if (
                other_token1.cov_ratio() > other_token2.cov_ratio()
            ):  # For more than 3 tokens, we should take max(other_tokens.cov_ratio)
                to_token = other_token1
                other = other_token2
            elif other_token1.cov_ratio() < other_token2.cov_ratio():
                to_token = other_token2
                other = other_token1
            else:  # when coverage ratio of other token 1 and token 2 are zero.
                force_swap = 0
                to_token = other_token1
                other = other_token2
                if print_action == True:
                    print("FORCE SWAP is not executed")

            self.swap_volatile_repeg(user, token, force_swap, to_token, other)

            if force_swap > 0:
                print(
                    f"FORCE SWAP ACTION: {force_swap} amount (with omega={omega}) of {force_swap} {token.name} to {other.name}"
                )
        # Withdraw_amount remain    #force_swap = 0 if force_swap=False, or
        withdraw_amount_remain = withdraw_amount - force_swap

        # cash transfer
        add_or_assign(token.name, user.wallet, withdraw_amount_remain)
        token.cash -= withdraw_amount_remain

        # token burning
        liability_burning = abs(liabilityToBurn)
        add_or_assign(token.name, user.lp_token, -abs(delta_lp))
        token.lp_token -= abs(delta_lp)
        token.liability -= abs(liability_burning)

        # withdrawal fees
        w_fees = liability_burning - withdraw_amount_remain
        w_fees_perc = w_fees / withdraw_amount_remain * 100

        # Price_adjustment after withdrawal:
        if deposit_withdraw_repeg == True:
            if (
                token.liability == 0
                or other_token1.liability == 0
                or other_token2.liability == 0
            ):
                return
            else:
                tweak_price(token, other_token1, other_token2)

        # Rounding negative decimal if -1<token.liability<0, #for absence of fee
        if -1 < token.liability < 0:
            token.liability = 0

        if Force_withdraw == True and force_swap > 0:
            if to_token is not None:  # Check if to_token is defined and not None
                print(
                    f"FORCE SWAP: {user.name} swap {force_swap} {token.name} to {to_token.name}"
                )
                print(
                    f"REMAIN WITHDRAWAL: The remaining amount of withdrawal is {withdraw_amount_remain} "
                )
        if print_action == True:
            print(
                f"WITHDRAW: {user.name} withdrew {liability_burning} {token.name} ({abs(delta_lp)} LP) and pay penalty of {w_fees} ({w_fees_perc}%) getting total {withdraw_amount} {token.name} assets out of the Pool"
            )
            print(f"withdraw amount is {withdraw_amount_remain}")
        return withdraw_amount_remain

    def swap_volatile_repeg(self, user, from_token, from_amount, to_token, other_token):
        if from_amount == 0 or from_amount == None:
            if print_update == True:
                print(
                    "!UNABLE TO SWAP: {0} has Empty balance for {1}".format(
                        user.name, from_token.name
                    )
                )
            return
        # For swapping negative amount
        if from_amount < 0:
            ft = to_token
            tt = from_token
        else:
            tt = to_token
            ft = from_token

        input_amount = abs(from_amount)
        if input_amount > user.wallet[ft.name]:
            input_amount = user.wallet[ft.name]

        scale_factor = price_factor(ft, tt)
        from_cash = ft.asset() * scale_factor
        from_liab = ft.liab() * scale_factor
        swap_amount = input_amount * scale_factor

        # Calculate the to_amount
        to_amount = swap_quote(
            from_cash, tt.asset(), from_liab, tt.liab(), swap_amount, A
        )

        # Set temporary variables
        ft.temp_swap = input_amount
        tt.temp_swap = -to_amount

        # dynamic Haircut
        dynamic_fee = calculate_fee(ft, tt)
        # print("Dynamic fee is {0}".format(dynamic_fee))
        h = dynamic_fee

        # we don't use USDT fee for repeg
        repeg_portion = repeg_ratio
        if to_token.name == "USDT":
            repeg_portion = 0

        # Retain the haircut in a way of depositing
        hf = (to_amount) * h  # h is the haircut rate
        hf_lp = (
            hf * (rrlp) * (1 - repeg_portion)
        )  # repeg_portion is the amount of lp put for repegging
        tt.hf_repeg += hf * (rrlp) * (repeg_portion)

        # Transfer to the user wallet
        act_to_amount = to_amount - hf

        add_or_assign(ft.name, user.wallet, -input_amount)
        ft.cash += input_amount

        # Logic, deduct to_amount first and add back the hf as deposit, so we can calculate the liability
        add_or_assign(tt.name, user.wallet, act_to_amount)
        tt.cash -= act_to_amount + hf

        if (
            -1 < tt.cash < 0
        ):  # Prevent deduction of zero asset out, Happened when there's no fee
            tt.cash = 0

        # Some haircut goes to LPs as a form of deposit
        hf_liab = mintfee(tt, hf_lp)
        tt.liability += hf_liab

        # Reset temporary variable
        ft.temp_swap = 0
        tt.temp_swap = 0

        # Record the system earning and culmulative haircut
        tt.hf_sys_records += hf * (1 - rrlp)
        tt.hf_lp_records += hf_liab

        if print_action == True:
            print(
                "SWAP: {0} swapped {1} {2} to {3} {4}. Fees charged: {5}".format(
                    user.name,
                    input_amount,
                    ft.name,
                    act_to_amount,
                    tt.name,
                    hf,
                )
            )
            print(
                f"The to_token {tt.name} has {tt.hf_repeg} haircut with its price last {tt.price_last} and price scale {tt.price_scale} "
            )

        # Repegging after the swap
        tweak_price(ft, tt, other_token)

        # Record the price repegging trajectory
        tokens_records(ft, tt, other_token)
        volume = (to_amount * tt.price_last) + (input_amount * ft.price_last) / 2

        ft.volume = volume
        tt.volume = volume
        other_token.volume = 0

        ft.volume_records.append(volume)
        tt.volume_records.append(volume)
        other_token.volume_records.append(0)
        tt.h_rate_records.append(h)
