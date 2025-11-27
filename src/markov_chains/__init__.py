"""
MarkovChains - A Python library for Markov Chain analysis in trading applications.

This library provides tools for building, analyzing, and applying Markov chains
to financial market data for algorithmic trading strategies.
"""

from .markov_chain import MarkovChain
from .trading_model import TradingMarkovModel, MarketState

__version__ = "0.1.0"
__all__ = ["MarkovChain", "TradingMarkovModel", "MarketState"]
