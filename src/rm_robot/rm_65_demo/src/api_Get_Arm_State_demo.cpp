//
// Created by ubuntu on 22-6-22.
//
#include <ros/ros.h>
#include <rm_msgs/Arm_Current_State.h>
#include <rm_msgs/GetArmState_Command.h>
# define PI 3.14159

// 接收到订阅的机械臂执行状态消息后，会进入消息回调函数
void GetArmState_Callback(const rm_msgs::Arm_Current_State msg)
{
    // 将接收到的消息打印出来，显示是否执行成功
    // ROS_INFO("joint state is: [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f]\n", msg.joint[0],msg.joint[1],msg.joint[2],msg.joint[3],msg.joint[4],msg.joint[5]);
    // ROS_INFO("pose state: [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f]\n", msg.Pose[0],msg.Pose[1],msg.Pose[2],msg.Pose[3],msg.Pose[4],msg.Pose[5]);
    ROS_INFO("x = %.2f, y = %.2f, z = %.2f\n", msg.Pose[0],msg.Pose[1],msg.Pose[2]);

    // ROS_INFO("rpy angle (PI): [%f, %f, %f]\n\n", msg.Pose[3]*180/PI, msg.Pose[4]*180/PI, msg.Pose[5]*180/PI);
    
    // ROS_INFO("arm_err is:%d\n", msg.arm_err);
    // ROS_INFO("sys_err is:%d\n", msg.sys_err);
}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "api_Get_Arm_State_demo");
    ros::NodeHandle nh;

    // 声明spinner对象，参数2表示并发线程数，默认处理全局Callback队列
    ros::AsyncSpinner spin(2);
    // 启动两个spinner线程并发执行可用回调 
    spin.start();

    ros::Duration(1.0).sleep();
    /*
     * 1.相关初始化
     */
    // 空间规划指令Publisher
    ros::Publisher test_Get_Arm_State_pub = nh.advertise<rm_msgs::GetArmState_Command>("/rm_driver/GetArmState_Cmd", 10);

    // 订阅机械臂执行状态话题
    ros::Subscriber planState_sub = nh.subscribe("/rm_driver/Arm_Current_State", 10, GetArmState_Callback);

 
     while (ros::ok())
    {
        // 发布空间规划指令以获取机械臂当前状态
        rm_msgs::GetArmState_Command command;
        command.command = "get_current_arm_state";
        test_Get_Arm_State_pub.publish(command);
        ROS_INFO("*******published command: %s", command.command.c_str());

        // 等待一段时间以避免过于频繁地发布
        ros::Duration(2.5).sleep();

        // 处理回调
        ros::spinOnce();
    }

    ros::waitForShutdown();

    return 0;
}
