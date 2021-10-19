# Created by wangjun 2021-10-12
# -*-coding: utf-8 -*-

import grpc
import proto.message_pb2 as message_pb2
import proto.message_pb2_grpc as message_pb2_grpc

def run():
    # connect the grpc server
    """
    localhost:50051
    """

    channel = grpc.insecure_channel('localhost:50051')

    # call rpc sever
    """
    MessageService就是生成的name_pdb2_grpc.py中的类class MessageService(object)，
    而Geeeter就是grpc协议文档helloword.proto中的service Greeter，然后拼接成GreeterStub

    stub.SayHello就是 helloword.proto中的rpc SayHello，
    helloword_pb2.HelloRequest就是 rpc SayHello(HelloRequest)中的请求"HelloRequest"，
    参数就是name='test'就是 string name=1 定义的参数名name，

    response.message获取响应的结果 就是请求体message HelloReply中定义的string message = 1参数名message
    """
    stub = message_pb2_grpc.MessageService(channel)
    # response = stub.SayHello(helloword_pb2.HelloRequest(name='test'))
    request_data = message_pb2.HelloRequest(name='test')
    response = stub.SayHello(request_data)

    print('Greeter client received: ' + response.message)


if __name__ == '__main__':
    run()