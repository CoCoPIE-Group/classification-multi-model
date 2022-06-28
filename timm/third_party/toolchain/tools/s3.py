"""
s3 tools
"""

import argparse
import boto3
from boto3.s3.transfer import TransferConfig

config = TransferConfig()
s3 = boto3.resource('s3')


# def upload_file(filename: str, bucket_name: str):
# 	key = fp = filename


def main():
	"""
	main function
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'-f', '--file', metavar='filename', type=str,
		action='store', help='File to be uploaded', required=True
	)
