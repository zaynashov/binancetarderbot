import ccxt
import pandas as pd
import requests
import json
import os
import time
import hmac
import hashlib
from binance import Client
from config import *
import win11toast as toast

exchange = ccxt.binance()

# блок номер 1 класы


class SymbolCycle:
    def __init__(self, filename):
        self.filename = filename
        self.symbols = self.load_symbols()
        self.current_index = 0

    def load_symbols(self):
        try:
            with open(self.filename, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            return []

    def save_symbols(self):
        with open(self.filename, 'w') as file:
            json.dump(self.symbols, file)

    def get_next_symbol(self):
        if not self.symbols:
            return None

        ip_symbol = self.symbols[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.symbols)
        self.save_symbols()
        return ip_symbol

# блок номер 2 взоимодействия с системой


def get_historical_data(symbol_ghd, timeframe):
    ohlcv = exchange.fetch_ohlcv(symbol_ghd, timeframe)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    file_name = f'{symbol_ghd}.csv'  # Генерация имени файла
    file_name = f'candles/{file_name}'
    df.to_csv(file_name, mode='w')  # Сохраняем полученные данные в файл
    addcurrentprices()


def addcurrentprices():
    # Открываем файл JSON
    with open("filtered_account_balance.json", "r") as json_file:
        data = json.load(json_file)

    # Извлекаем массив positions
    positions = data["positions"]

    if positions:
        for position in positions:
            # Извлекаем значение symbol для каждой позиции
            cpsymbol = position["symbol"]
            unrealizedprofit = float(position["unrealizedProfit"])  # Преобразуем в float
            entryprice = float(position["entryPrice"])  # Преобразуем в float
            # positionside = position["positionSide"]

            # Делаем запрос к бирже Binance для получения текущей цены токена
            binance_api_url = f"https://api.binance.com/api/v3/ticker/price?symbol={cpsymbol}"
            response = requests.get(binance_api_url)
            price_data = response.json()

            # Извлекаем цену из ответа
            current_price = float(price_data.get("price"))  # Преобразуем в float

            # Добавляем новое значение currentPrice к текущей позиции
            position["currentPrice"] = current_price
            if current_price > entryprice and unrealizedprofit > 0:
                position["positionSide"] = "LONG"
            elif current_price < entryprice and unrealizedprofit < 0:
                position["positionSide"] = "LONG"
            elif current_price > entryprice and unrealizedprofit < 0:
                position["positionSide"] = "SHORT"
            elif current_price < entryprice and unrealizedprofit > 0:
                position["positionSide"] = "SHORT"

        # Записываем обновленные данные в файл JSON
        with open("filtered_account_balance.json", "w") as json_file:
            json.dump(data, json_file, indent=2)

        # Выводим информацию для проверки
        # print("Added Current Prices and Updated Position Sides in positions.")
    else:
        pass


def get_filtered_account_data(gfad_api_key, gfad_secret_key):
    def generate_signature(gs_query_string, gs_secret_key):
        return hmac.new(gs_secret_key.encode('utf-8'), gs_query_string.encode('utf-8'), hashlib.sha256).hexdigest()

    base_url = 'https://fapi.binance.com'
    endpoint = '/fapi/v2/account'

    timestamp = int(time.time() * 1000)
    gfad_query_string = f'timestamp={timestamp}'
    signature = generate_signature(gfad_query_string, gfad_secret_key)

    headers = {
        'X-MBX-APIKEY': gfad_api_key
    }
    params = {
        'timestamp': timestamp,
        'signature': signature
    }

    response = requests.get(base_url + endpoint, headers=headers, params=params)
    account_balance = response.json()

    # Фильтрация активов
    filtered_assets = [asset for asset in account_balance['assets'] if asset['asset'] == 'USDT']

    # Фильтрация позиций
    filtered_positions = (
        [position for position in account_balance['positions'] if abs(float(position['positionAmt'])) > 0]
    )

    # Создание словаря для сохранения
    filtered_data = {
        'assets': filtered_assets,
        'positions': filtered_positions
    }

    # Проверка существования файла и сохранение
    file_name = 'filtered_account_balance.json'
    if os.path.exists(file_name):
        with open(file_name, 'w') as json_file:
            json.dump(filtered_data, json_file, indent=2)
        # print(f"Файл '{file_name}' был перезаписан.")
    else:
        with open(file_name, 'w') as json_file:
            json.dump(filtered_data, json_file, indent=2)
        print(f"Файл '{file_name}' был создан.")

    addcurrentprices()

    return filtered_data


def acc_trade_check():  # проверка аккаунта торгуется или нет в данный момент
    with open("filtered_account_balance.json", "r") as json_file:
        data = json.load(json_file)

    # Проверьте, есть ли массив assets и что initialMargin равно нулю в любом из его элементов
    if "assets" in data:
        for asset in data["assets"]:
            if "initialMargin" in asset and float(asset["initialMargin"]) > 0:
                inpositions_atc = True
            else:
                inpositions_atc = False
            return inpositions_atc


def acc_trade_position():  # проверка аккаунта торгуется или нет в данный момент
    with open("filtered_account_balance.json", "r") as json_file:
        data = json.load(json_file)

    # Проверьте, есть ли массив assets и что initialMargin равно нулю в любом из его элементов
    if "positions" in data:
        for positions in data["positions"]:
            positionside_atp = positions["positionSide"]
            positionentryprice_atp = positions["entryPrice"]
            return positionside_atp, positionentryprice_atp


def acc_trade_pnl():  # проверка аккаунта торгуется или нет в данный момент
    with open("filtered_account_balance.json", "r") as json_file:
        data = json.load(json_file)

    # Проверьте, есть ли массив assets и что initialMargin равно нулю в любом из его элементов
    if "positions" in data:
        for positions in data["positions"]:
            positionpnl_atp = float(positions["unrealizedProfit"])
            positionpnl_atp = f"{positionpnl_atp:.2f}"
            return positionpnl_atp


# блок номер 3 индикаторы
def strategy_atr(data_file, sa_atr_length, sa_atr_period, sa_ema_length):
    data_file = f'candles/{data_file}'
    global signal
    try:
        df = pd.read_csv(data_file, parse_dates=['timestamp'], index_col='timestamp')

        tr = pd.DataFrame({'h-l': df['high'] - df['low'], 'h-pc': abs(df['high'] - df['close'].shift()),
                           'l-pc': abs(df['low'] - df['close'].shift())})
        tr['tr'] = tr[['h-l', 'h-pc', 'l-pc']].max(axis=1, skipna=False)
        atr = tr['tr'].rolling(window=sa_atr_length).mean()

        df['ha-close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        df['ha-open'] = (df['open'].shift() + df['close'].shift()) / 2
        df['ha-high'] = df[['high', 'ha-open', 'ha-close']].max(axis=1, skipna=False)
        df['ha-low'] = df[['low', 'ha-open', 'ha-close']].min(axis=1, skipna=False)
        df['ema'] = df['ha-close'].ewm(span=sa_ema_length).mean()
        df['long_trail'] = df['ha-high'] - atr * sa_atr_period
        df['short_trail'] = df['ha-low'] + atr * sa_atr_period

        if df['ema'].iloc[-1] > df['ema'].iloc[-2]:
            trend = 'buy'
        else:
            trend = 'sell'

        if trend == 'buy':
            if df['ha-close'].iloc[-1] > df['long_trail'].iloc[-1]:
                if trend != signal:
                    signal = trend
                    return signal
        elif trend == 'sell':
            if df['ha-close'].iloc[-1] < df['short_trail'].iloc[-1]:
                if trend != signal:
                    signal = trend
                    return signal

    except Exception as e:
        print(e)


def strategy_rsi(data_file):
    data_file = f'candles/{data_file}'

    def calculate_rsi(data):
        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=rsi_length).mean()
        avg_loss = loss.rolling(window=rsi_length).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    try:
        df = pd.read_csv(data_file, parse_dates=['timestamp'], index_col='timestamp')

        rsi_data = df[['close']].copy()
        rsi_data['rsi'] = calculate_rsi(rsi_data)

        last_rsi = rsi_data['rsi'].iloc[-1]
        prev_rsi = rsi_data['rsi'].iloc[-5]

        if last_rsi > prev_rsi:
            direction = 'Bullish'
        elif last_rsi < prev_rsi:
            direction = 'Bearish'
        else:
            direction = 'NONE'

        return last_rsi, direction

    except Exception as e:
        print(f"Error: {e}")
        pass


def strategy_macd(data_file):
    def calculate_macd(data, short_period=12, long_period=26, signal_period=9):
        # Вычисление короткой и длинной экспоненциальной скользящей средней (EMA)
        ema_short = data['close'].ewm(span=short_period).mean()
        ema_long = data['close'].ewm(span=long_period).mean()

        # Расчет MACD
        cal_macd = ema_short - ema_long

        # Расчет сигнальной линии (EMA от MACD)
        cal_signal_line = cal_macd.ewm(span=signal_period).mean()

        # Расчет гистограммы (разница между MACD и сигнальной линией)
        cal_histogram = cal_macd - cal_signal_line

        return cal_macd, cal_signal_line, cal_histogram
    # Загрузка данных из файла (замените на свой путь к файлу)
    data_file = f'candles/{data_file}'
    df = pd.read_csv(data_file, parse_dates=['timestamp'], index_col='timestamp')

    # Вызываем функцию расчета MACD
    m_macd, signal_line, histogram = calculate_macd(df)

    # Определение направления индикатора MACD
    m_macd = m_macd.iloc[-1]
    m_signal_line = signal_line.iloc[-1]

    if m_macd > m_signal_line:
        md_direction = 'Bullish'  # Если MACD выше сигнальной линии, это Bullish (восходящий тренд)
    elif m_macd < m_signal_line:
        md_direction = 'Bearish'  # Если MACD ниже сигнальной линии, это Bearish (нисходящий тренд)
    else:
        md_direction = 'NONE'
    return m_macd, m_signal_line, md_direction


# блок номер 5 вычисления
def calculate_roe():
    def calc_roe(cal_deposit, cal_leverage, cal_entry_price, cal_exit_price):
        try:
            cal_deposit = float(cal_deposit)
            cal_leverage = float(cal_leverage)
            cal_entry_price = float(cal_entry_price)
            cal_exit_price = float(cal_exit_price)
        except ValueError:
            print("Ошибка при преобразовании данных в числа")
            return None

        trade_size = cal_deposit * cal_leverage
        num_shares = (trade_size / cal_entry_price)
        revenue = num_shares * cal_exit_price
        cal_roe = revenue - trade_size
        percent_roe = cal_roe / cal_deposit * 100
        cal_roe = f"{percent_roe:.2f}"
        return cal_roe
    while True:
        with open("filtered_account_balance.json", "r") as json_file:
            data = json.load(json_file)
        positions = data["positions"]
        if positions:
            for position in positions:
                roe_symbol = position["symbol"]
                # positionside = position["positionside"]
                positioninitialmargin = float(position["positionInitialMargin"])  # Преобразуем в float
                entryprice = float(position["entryPrice"])  # Преобразуем в float
                leverage_roe = position["leverage"]
                binance_api_url = f"https://api.binance.com/api/v3/ticker/price?symbol={roe_symbol}"
                response = requests.get(binance_api_url)
                price_data = response.json()
                current_price = float(price_data.get("price"))  # Преобразуем в float
                roe2 = calc_roe(positioninitialmargin, leverage_roe, entryprice, current_price)
                time.sleep(0.01)
                return roe_symbol, roe2


# блок номер 6, запросы


def get_token_price(symbol_gtp):
    exchange_gtp = ccxt.binance()  # Инициализация биржевого объекта (в данном случае Binance)

    try:
        ticker = exchange_gtp.fetch_ticker(symbol_gtp)  # Получение данных по торговой паре
        if 'last' in ticker:
            token_price = float(ticker['last'])  # Получение последней цены
            return token_price
        else:
            return None
    except Exception as e:
        print(f"Ошибка при получении цены с биржи: {e}")
        return None


def calculate_token_amount(symbol_cta, available_funds, leverage_cta):
    token_price = get_token_price(symbol_cta)
    try:
        available_funds = float(available_funds)
        lvrg = float(leverage_cta)
        token_price = float(token_price)
    except ValueError:
        print("Ошибка при преобразовании данных в числа")
        return None

    if lvrg <= 0:
        print("Кредитное плечо должно быть положительным числом")
        return None

    if token_price <= 0:
        print("Цена токена должна быть положительной")
        return None

    # Вычисляем количество токенов
    token_amount_cta = (available_funds * lvrg) / token_price

    return token_amount_cta


def format_quantity(symbol_fq, quantity_fq):
    exchange_info = client.futures_exchange_info()
    symbol_info = next(item for item in exchange_info['symbols'] if item['symbol'] == symbol_fq)
    precision = symbol_info['quantityPrecision']
    return f"{quantity_fq:.{precision}f}"


# ордера


def new_position(symbol_np, side_np, quantity_np):
    client.futures_create_order(symbol=symbol_np, side=side_np, type='MARKET', quantity=quantity_np)


def close_position_market(symbol_cpm):
    position_info = client.futures_position_information(symbol=symbol_cpm)
    positionamt_cpm = float(position_info[0]['positionAmt'])

    if positionamt_cpm > 0:
        position_side = 'LONG'
    elif positionamt_cpm < 0:
        position_side = 'SHORT'
    else:
        position_side = 'NONE'

    if position_side == 'LONG':
        client.futures_create_order(
            symbol=symbol_cpm,
            side='SELL', type='MARKET',
            quantity=abs(positionamt_cpm)
        )
    elif position_side == 'SHORT':
        client.futures_create_order(symbol=symbol_cpm,
                                    side='BUY',
                                    type='MARKET',
                                    quantity=abs(positionamt_cpm)
                                    )
    else:
        print('Error')


def position_take_profi(symbol_tpf, amount_percent):
    position_info = client.futures_position_information(symbol=symbol_tpf)
    positionamt_tpf = float(position_info[0]['positionAmt'])

    if positionamt_tpf > 0:
        position_side = 'LONG'
    elif positionamt_tpf < 0:
        position_side = 'SHORT'
    else:
        position_side = 'NONE'
    amount_percent = amount_percent / 100
    positionamt_tpf = positionamt_tpf * amount_percent
    positionamount_tpf = float(format_quantity(symbol_tpf, positionamt_tpf))

    if position_side == 'LONG':
        client.futures_create_order(
            symbol=symbol_tpf,
            side='SELL', type='MARKET',
            quantity=abs(positionamount_tpf)
        )
    elif position_side == 'SHORT':
        client.futures_create_order(symbol=symbol_tpf,
                                    side='BUY',
                                    type='MARKET',
                                    quantity=abs(positionamount_tpf)
                                    )
    else:
        print('Error')


# блок номер 4 определение сигнала


def define_signal(symbol_ds):
    ds_signal = 'signal'
    atr_signal = strategy_atr(f'{symbol_ds}.csv', atr_length, atr_period, ema_length)
    rsi_ds, rsi_direction = strategy_rsi(f'{symbol_ds}.csv')
    macd_ds, macd_signal_line, macd_direction = strategy_macd(f'{symbol_ds}.csv')
    if atr_signal != ds_signal:
        ds_signal = atr_signal
    if ds_signal == 'sell' and \
            macd_direction == 'Bearish' and \
            rsi_direction == 'Bearish':
        out_signal = 'SHORT'
    elif ds_signal == 'buy' and \
            macd_direction == 'Bullish' and \
            rsi_direction == 'Bullish':
        out_signal = 'LONG'
    else:
        out_signal = 'NONE'
    return out_signal


# основной блок, начальная точка выполнения кода
if __name__ == "__main__":
    api_key = api
    api_secret = secret
    client = Client(api_key, api_secret)
    bot_timeframe = '5m'  # только стандартные значения: 1m, 5m, 15m, 30m, 1h, 2h
    atr_length = 2
    atr_period = 10
    ema_length = 50
    rsi_length = 14  # период для расчета RSI
    target_rsi_high = 75  # функция даст положительный сигнал если RSI ниже этого значения
    target_rsi_low = 25  # функция даст положительный сигнал если RSI выше этого значения
    signal = 'signal'
    cicle = 0
    inposition = False
    take_profit_1 = False
    take_profit_2 = False
    take_profit_3 = False
    symbol_cycle = SymbolCycle('pairs.json')
    quantity_usdt = 5
    leverage = 20
    while True:
        get_filtered_account_data(api_key, api_secret)
        addcurrentprices()
        time.sleep(0.01)
        intrade = acc_trade_check()
        if not intrade:
            take_profit_1 = False
            take_profit_2 = False
            take_profit_3 = False
            while True:
                symbol = symbol_cycle.get_next_symbol()
                get_historical_data(symbol, bot_timeframe)
                # bnce_symbol = symbol.replace("/", "")
                signal = define_signal(symbol)
                print(f'Сигнал пары {symbol}: {signal}')
                if signal != 'NONE':
                    space = '  '
                    print(f"{symbol}: signal:{signal}")
                    toast.toast('Новый сигнал', symbol + space + signal)
                if symbol == 'CYBERUSDT':
                    cicle = cicle + 1
                    print(f'Цикл {cicle} закончен, ждем следующего цикла')
                    if cicle % 10 == 0:
                        print("Счетчик кратен 10. Ожидание 10 минуту...")
                        time.sleep(60 * 10)  # Подождать 10 минут
                    else:
                        time.sleep(60 * 2)
                if signal == 'LONG':
                    quantity = calculate_token_amount(symbol, quantity_usdt, leverage)
                    token_amount = format_quantity(symbol, quantity)
                    new_position(symbol, side_np='BUY', quantity_np=token_amount)
                    get_filtered_account_data(api_key, api_secret)
                    intrade = True
                    break
                if signal == 'SHORT':
                    quantity = calculate_token_amount(symbol, quantity_usdt, leverage)
                    token_amount = format_quantity(symbol, quantity)
                    new_position(symbol, side_np='SELL', quantity_np=token_amount)
                    get_filtered_account_data(api_key, api_secret)
                    intrade = True
                    break
                time.sleep(0.1)
        if intrade:
            while True:
                get_filtered_account_data(api_key, api_secret)
                intrade = acc_trade_check()
                if not intrade:
                    break
                symbol_trade, roe_trade_inposition = calculate_roe()
                positionside_trade, positionentryprice_trade = acc_trade_position()
                mark_price = get_token_price(symbol_trade)
                get_historical_data(symbol_trade, bot_timeframe)
                atr_signal_trade = strategy_atr(f'{symbol_trade}.csv', atr_length, atr_period, ema_length)
                pnl_trade = acc_trade_pnl()
                # print(atr_signal_trade)
                # print(f'roe пряиой запрос {roe_trade_inposition}')
                roe_trade = abs(float(roe_trade_inposition))
                # print(f'roe после roe_trade = abs(float(roe_trade_inposition)){roe_trade}')
                if positionside_trade == "LONG" and float(mark_price) < float(positionentryprice_trade):
                    roe_trade = roe_trade * -1
                if positionside_trade == "SHORT" and float(mark_price) > float(positionentryprice_trade):
                    roe_trade = roe_trade * -1
                print(f'{symbol_trade}: {positionside_trade},  Нереализованная PNL:  {pnl_trade}  {roe_trade}%  ')
                if positionside_trade == "LONG" and atr_signal_trade == 'sell':
                    close_position_market(symbol_trade)
                    toast.toast('Position closed', f'Позиция {symbol_trade} \
                    закрыта из-за изменения сигнала в отрицательную сторону')
                    get_filtered_account_data(api_key, api_secret)
                    break
                if positionside_trade == "SHORT" and atr_signal_trade == 'buy':
                    close_position_market(symbol_trade)
                    toast.toast('Position closed', f'Позиция {symbol_trade} \
                    закрыта из-за изменения сигнала в положительную сторону')
                    get_filtered_account_data(api_key, api_secret)
                    break
                if roe_trade < 0:
                    if abs(roe_trade) > 30:
                        close_position_market(symbol_trade)
                        toast.toast('Position closed', f'Позиция {symbol_trade} закрыта в минус')
                        get_filtered_account_data(api_key, api_secret)
                        break
                if roe_trade > 0:
                    if roe_trade > 15 and not take_profit_1:
                        position_take_profi(symbol_trade, amount_percent=50)
                        toast.toast('Take profit', f'Тейк профит 1 {symbol_trade}')
                        take_profit_1 = True
                        get_filtered_account_data(api_key, api_secret)
                    if roe_trade > 25 and take_profit_1 and not take_profit_2:
                        position_take_profi(symbol_trade, amount_percent=30)
                        toast.toast('Take profit', f'Тейк профит 2 {symbol_trade}')
                        take_profit_2 = True
                        get_filtered_account_data(api_key, api_secret)
                    if roe_trade > 45 and take_profit_1 and take_profit_2 and not take_profit_3:
                        position_take_profi(symbol_trade, amount_percent=30)
                        toast.toast('Position closed', f'Позиция {symbol_trade} закрыта с прыбылью')
                        get_filtered_account_data(api_key, api_secret)
                        take_profit_3 = True
                    if take_profit_1 and roe_trade < 3:
                        close_position_market(symbol_trade)
                        toast.toast('Position closed', f'Позиция {symbol_trade} \
                                            закрыта по трейлинг стопу')
                        break
                    if take_profit_2 and roe_trade < 10:
                        close_position_market(symbol_trade)
                        toast.toast('Position closed', f'Позиция {symbol_trade} \
                                            закрыта по трейлинг стопу')
                        break
                    if take_profit_3 and roe_trade < 25:
                        close_position_market(symbol_trade)
                        toast.toast('Position closed', f'Позиция {symbol_trade} \
                                            закрыта по трейлинг стопу')
                        break
                    if roe_trade > 70:
                        close_position_market(symbol_trade)
                        break
                time.sleep(0.01)
