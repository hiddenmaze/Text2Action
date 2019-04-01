function [] = plot_dummy_3d( plot_pred_3d, fig_num )

head_radius = 0.32;
joint_radius = 0.1;
neck_radius = 0.15;
arm_thick = 0.1;

arm_color = [229 206 178]/255;
body_color = [240 215 186]/255;
joint_color = [211 186 141]/255;
edge_color = [168 136 87]/255;

[sph_x, sph_y, sph_z] = sphere(8);
head_x = head_radius * sph_x;
head_y = head_radius * sph_y;
head_z = head_radius * sph_z;

joint_x = joint_radius * sph_x;
joint_y = joint_radius * sph_y;
joint_z = joint_radius * sph_z;

neck_x = neck_radius * sph_x;
neck_y = neck_radius * sph_y;
neck_z = neck_radius * sph_z;

body_len = 2;

figure(fig_num);
clf;
hold on;
axis('equal',[-3 3 -3 3 -3 3]);
view([65, 5]);

lwrist = patch(surf2patch(zeros(6), zeros(6),zeros(6)));
set(lwrist,'EdgeColor',edge_color,'EdgeAlpha',0.2);
set(lwrist, 'FaceColor', arm_color);
      
rwrist = patch(surf2patch(zeros(6), zeros(6),zeros(6)));
set(rwrist,'EdgeColor',edge_color,'EdgeAlpha',0.2);
set(rwrist, 'FaceColor', arm_color);

lforearm = patch(surf2patch(zeros(6), zeros(6),zeros(6)));
set(lforearm,'EdgeColor',edge_color,'EdgeAlpha',0.2);
set(lforearm, 'FaceColor', arm_color);

rforearm = patch(surf2patch(zeros(6), zeros(6),zeros(6)));
set(rforearm,'EdgeColor',edge_color,'EdgeAlpha',0.2);
set(rforearm, 'FaceColor', arm_color);

lshoulder = patch(surf2patch(zeros(6), zeros(6),zeros(6)));
set(lshoulder,'EdgeColor',edge_color,'EdgeAlpha',0.2);
set(lshoulder, 'FaceColor', arm_color);

rshoulder = patch(surf2patch(zeros(6), zeros(6),zeros(6)));
set(rshoulder,'EdgeColor',edge_color,'EdgeAlpha',0.2);
set(rshoulder, 'FaceColor', arm_color);

head = patch(surf2patch(zeros(8), zeros(8),zeros(8)));
set(head, 'EdgeColor', edge_color, 'EdgeAlpha', 0.2);
set(head, 'FaceColor', body_color);
    
body = fill3([0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0], body_color, 'edgecolor', 'none' );

lshoulder_joint = patch(surf2patch(zeros(8), zeros(8),zeros(8)));
set(lshoulder_joint, 'EdgeColor', edge_color, 'EdgeAlpha', 0.2);
set(lshoulder_joint, 'FaceColor', joint_color);

rshoulder_joint = patch(surf2patch(zeros(8), zeros(8),zeros(8)));
set(rshoulder_joint, 'EdgeColor', edge_color, 'EdgeAlpha', 0.2);
set(rshoulder_joint, 'FaceColor', joint_color);

neck_joint = patch(surf2patch(zeros(8), zeros(8),zeros(8)));
set(neck_joint, 'EdgeColor', edge_color, 'EdgeAlpha', 0.2);
set(neck_joint, 'FaceColor', joint_color);

lelbow = patch(surf2patch(zeros(8), zeros(8),zeros(8)));
set(lelbow, 'EdgeColor', edge_color, 'EdgeAlpha', 0.2);
set(lelbow, 'FaceColor',  joint_color);

relbow = patch(surf2patch(zeros(8), zeros(8),zeros(8)));
set(relbow, 'EdgeColor', edge_color, 'EdgeAlpha', 0.2);
set(relbow, 'FaceColor',  joint_color);

lhand = patch(surf2patch(zeros(8), zeros(8),zeros(8)));
set(lhand, 'EdgeColor', edge_color, 'EdgeAlpha', 0.2);
set(lhand, 'FaceColor', joint_color);

rhand = patch(surf2patch(zeros(8), zeros(8),zeros(8)));
set(rhand, 'EdgeColor', edge_color, 'EdgeAlpha', 0.2);
set(rhand, 'FaceColor', joint_color);

for i= 1:size(plot_pred_3d, 3)
    tic;
    body_vector = (plot_pred_3d(:, 3, i) + plot_pred_3d(:, 6, i)) / 2 - plot_pred_3d(:, 2, i);
    body_bottom =plot_pred_3d(:, 2, i) +  body_len * body_vector / norm(body_vector);    
    
    lwrist_center = (plot_pred_3d(:, 4, i) + plot_pred_3d(:, 5, i) ) /2;
    lwrist_len = norm(plot_pred_3d(:, 4, i) - plot_pred_3d(:, 5, i));                
    lwrist_vec = plot_pred_3d(:, 5, i) - plot_pred_3d(:, 4, i);
    lwrist_U = [null(plot_pred_3d(:, 4, i)' - plot_pred_3d(:, 5, i)')' ; lwrist_vec'/norm(lwrist_vec)];
    
    rwrist_center = (plot_pred_3d(:, 7, i) + plot_pred_3d(:, 8, i) ) /2;
    rwrist_len = norm(plot_pred_3d(:, 7, i) - plot_pred_3d(:, 8, i));                
    rwrist_vec = plot_pred_3d(:, 8, i) - plot_pred_3d(:, 7, i);
    rwrist_U = [null(plot_pred_3d(:, 7, i)' - plot_pred_3d(:, 8, i)')' ; rwrist_vec'/norm(rwrist_vec)];    
    
    lforearm_center = (plot_pred_3d(:, 3, i) + plot_pred_3d(:, 4, i) ) /2;
    lforearm_len = norm(plot_pred_3d(:, 3, i) - plot_pred_3d(:, 4, i));                
    lforearm_vec = plot_pred_3d(:, 4, i) - plot_pred_3d(:, 3, i);
    lforearm_U = [null(plot_pred_3d(:, 3, i)' - plot_pred_3d(:, 4, i)')' ; lforearm_vec'/norm(lforearm_vec)];
    
    rforearm_center = (plot_pred_3d(:, 6, i) + plot_pred_3d(:, 7, i) ) /2;
    rforearm_len = norm(plot_pred_3d(:, 6, i) - plot_pred_3d(:, 7, i));                
    rforearm_vec = plot_pred_3d(:, 7, i) - plot_pred_3d(:, 6, i);
    rforearm_U = [null(plot_pred_3d(:, 6, i)' - plot_pred_3d(:, 7, i)')' ; rforearm_vec'/norm(rforearm_vec)];
   
    lshoulder_center = (plot_pred_3d(:, 2, i) + plot_pred_3d(:, 3, i) ) /2;
    lshoulder_len = norm(plot_pred_3d(:, 2, i) - plot_pred_3d(:, 3, i));                
    lshoulder_vec = plot_pred_3d(:, 3, i) - plot_pred_3d(:, 2, i);
    lshoulder_U = [null(plot_pred_3d(:, 2, i)' - plot_pred_3d(:, 3, i)')' ; lshoulder_vec'/norm(lshoulder_vec)];
    
    rshoulder_center = (plot_pred_3d(:, 2, i) + plot_pred_3d(:, 6, i) ) /2;
    rshoulder_len = norm(plot_pred_3d(:, 2, i) - plot_pred_3d(:, 6, i));                
    rshoulder_vec = plot_pred_3d(:, 6, i) - plot_pred_3d(:, 2, i);
    rshoulder_U = [null(plot_pred_3d(:, 2, i)' - plot_pred_3d(:, 6, i)')' ; rshoulder_vec'/norm(rshoulder_vec)];   

    
    % wrist
    [lwrist_x, lwrist_y, lwrist_z] = ellipsoid(0, 0, 0, arm_thick, arm_thick, lwrist_len/2, 6);
    lwrist_newXYZ = lwrist_U'*[lwrist_x(:)';lwrist_y(:)'; lwrist_z(:)'];
    lwrist_xnew = reshape(lwrist_newXYZ(1,:)+lwrist_center(1),size(lwrist_x,1),size(lwrist_x,2));
    lwrist_ynew = reshape(lwrist_newXYZ(2,:)+lwrist_center(2),size(lwrist_x,1),size(lwrist_x,2));
    lwrist_znew = reshape(lwrist_newXYZ(3,:)+lwrist_center(3),size(lwrist_x,1),size(lwrist_x,2));
   
    lwrist_patch = surf2patch(lwrist_xnew,lwrist_ynew,lwrist_znew);
    set(lwrist, 'Faces', lwrist_patch.faces, 'Vertices', lwrist_patch.vertices);

    [rwrist_x, rwrist_y, rwrist_z] = ellipsoid(0, 0, 0, arm_thick, arm_thick, rwrist_len/2, 6);
    rwrist_newXYZ = rwrist_U'*[rwrist_x(:)';rwrist_y(:)'; rwrist_z(:)'];
    rwrist_xnew = reshape(rwrist_newXYZ(1,:)+rwrist_center(1),size(rwrist_x,1),size(rwrist_x,2));
    rwrist_ynew = reshape(rwrist_newXYZ(2,:)+rwrist_center(2),size(rwrist_x,1),size(rwrist_x,2));
    rwrist_znew = reshape(rwrist_newXYZ(3,:)+rwrist_center(3),size(rwrist_x,1),size(rwrist_x,2));

    rwrist_patch = surf2patch(rwrist_xnew,rwrist_ynew,rwrist_znew);
    set(rwrist, 'Faces', rwrist_patch.faces, 'Vertices', rwrist_patch.vertices);
   
    % forearm
    
    [lforearm_x, lforearm_y, lforearm_z] = ellipsoid(0, 0, 0, arm_thick, arm_thick, lforearm_len/2, 6);
    lforearm_newXYZ = lforearm_U'*[lforearm_x(:)';lforearm_y(:)'; lforearm_z(:)'];
    lforearm_xnew = reshape(lforearm_newXYZ(1,:)+lforearm_center(1),size(lforearm_x,1),size(lforearm_x,2));
    lforearm_ynew = reshape(lforearm_newXYZ(2,:)+lforearm_center(2),size(lforearm_x,1),size(lforearm_x,2));
    lforearm_znew = reshape(lforearm_newXYZ(3,:)+lforearm_center(3),size(lforearm_x,1),size(lforearm_x,2));
       
    lforearm_patch = surf2patch(lforearm_xnew, lforearm_ynew, lforearm_znew);
    set(lforearm, 'Faces', lforearm_patch.faces, 'Vertices', lforearm_patch.vertices);
    
    [rforearm_x, rforearm_y, rforearm_z] = ellipsoid(0, 0, 0, arm_thick, arm_thick, rforearm_len/2, 6);
    rforearm_newXYZ = rforearm_U'*[rforearm_x(:)';rforearm_y(:)'; rforearm_z(:)'];
    rforearm_xnew = reshape(rforearm_newXYZ(1,:)+rforearm_center(1),size(rforearm_x,1),size(rforearm_x,2));
    rforearm_ynew = reshape(rforearm_newXYZ(2,:)+rforearm_center(2),size(rforearm_x,1),size(rforearm_x,2));
    rforearm_znew = reshape(rforearm_newXYZ(3,:)+rforearm_center(3),size(rforearm_x,1),size(rforearm_x,2));
    
    rforearm_patch = surf2patch(rforearm_xnew, rforearm_ynew, rforearm_znew);
    set(rforearm, 'Faces', rforearm_patch.faces, 'Vertices', rforearm_patch.vertices);
    
    % shoulder
    
    [lshoulder_x, lshoulder_y, lshoulder_z] = ellipsoid(0, 0, 0, arm_thick, arm_thick, lshoulder_len/2, 6);
    lshoulder_newXYZ = lshoulder_U'*[lshoulder_x(:)';lshoulder_y(:)'; lshoulder_z(:)'];
    lshoulder_xnew = reshape(lshoulder_newXYZ(1,:)+lshoulder_center(1),size(lshoulder_x,1),size(lshoulder_x,2));
    lshoulder_ynew = reshape(lshoulder_newXYZ(2,:)+lshoulder_center(2),size(lshoulder_x,1),size(lshoulder_x,2));
    lshoulder_znew = reshape(lshoulder_newXYZ(3,:)+lshoulder_center(3),size(lshoulder_x,1),size(lshoulder_x,2));
   
    lshoulder_patch = surf2patch(lshoulder_xnew,lshoulder_ynew,lshoulder_znew);
    set(lshoulder, 'Faces', lshoulder_patch.faces, 'Vertices', lshoulder_patch.vertices);
            
   [rshoulder_x, rshoulder_y, rshoulder_z] = ellipsoid(0, 0, 0, arm_thick, arm_thick, rshoulder_len/2, 6);
    rshoulder_newXYZ = rshoulder_U'*[rshoulder_x(:)';rshoulder_y(:)'; rshoulder_z(:)'];
    rshoulder_xnew = reshape(rshoulder_newXYZ(1,:)+rshoulder_center(1),size(rshoulder_x,1),size(rshoulder_x,2));
    rshoulder_ynew = reshape(rshoulder_newXYZ(2,:)+rshoulder_center(2),size(rshoulder_x,1),size(rshoulder_x,2));
    rshoulder_znew = reshape(rshoulder_newXYZ(3,:)+rshoulder_center(3),size(rshoulder_x,1),size(rshoulder_x,2));
   
    rshoulder_patch = surf2patch(rshoulder_xnew,rshoulder_ynew,rshoulder_znew);
    set(rshoulder, 'Faces', rshoulder_patch.faces, 'Vertices', rshoulder_patch.vertices);
    
    % head
    head_patch = surf2patch(head_x+plot_pred_3d(1, 1, i), head_y+plot_pred_3d(2, 1, i),...
                    head_z+plot_pred_3d(3, 1, i));
    set(head, 'Faces', head_patch.faces, 'Vertices', head_patch.vertices);

    % body
    set(body, 'XData', [plot_pred_3d(1, 2, i), plot_pred_3d(1, 3, i), body_bottom(1), plot_pred_3d(1, 6, i)],...
        'YData', [plot_pred_3d(2, 2, i), plot_pred_3d(2, 3, i), body_bottom(2), plot_pred_3d(2, 6, i)],...
        'ZData', [plot_pred_3d(3, 2, i), plot_pred_3d(3, 3, i), body_bottom(3), plot_pred_3d(3, 6, i)]);

   % shoulder
    lshoulder_joint_patch = surf2patch(joint_x+plot_pred_3d(1, 3, i), joint_y+plot_pred_3d(2, 3, i),...
                        joint_z+plot_pred_3d(3, 3, i));
    set(lshoulder_joint, 'Faces', lshoulder_joint_patch.faces, 'Vertices', lshoulder_joint_patch.vertices);

    rshoulder_joint_patch = surf2patch(joint_x+plot_pred_3d(1, 6, i), joint_y+plot_pred_3d(2, 6, i),...
                        joint_z+plot_pred_3d(3, 6, i));
    set(rshoulder_joint, 'Faces', rshoulder_joint_patch.faces, 'Vertices', rshoulder_joint_patch.vertices);

    % neck    
    neck_joint_patch = surf2patch(neck_x+plot_pred_3d(1, 2, i), neck_y+plot_pred_3d(2, 2, i),...
                        neck_z+plot_pred_3d(3, 2, i));
    set(neck_joint, 'Faces', neck_joint_patch.faces, 'Vertices', neck_joint_patch.vertices);

    % elbow
    lelbow_patch = surf2patch(joint_x+plot_pred_3d(1, 4, i), joint_y+plot_pred_3d(2, 4, i),...
                        joint_z+plot_pred_3d(3, 4, i));
    set(lelbow, 'Faces', lelbow_patch.faces, 'Vertices', lelbow_patch.vertices);  
    
    relbow_patch = surf2patch(joint_x+plot_pred_3d(1, 7, i), joint_y+plot_pred_3d(2, 7, i),...
                        joint_z+plot_pred_3d(3, 7, i));
    set(relbow, 'Faces', relbow_patch.faces, 'Vertices', relbow_patch.vertices);       

    % hand    

    lhand_patch =surf2patch(joint_x+plot_pred_3d(1, 5, i), joint_y+plot_pred_3d(2, 5, i),...
                        joint_z+plot_pred_3d(3, 5, i));
    set(lhand, 'Faces', lhand_patch.faces, 'Vertices', lhand_patch.vertices);
    
    rhand_patch =surf2patch(joint_x+plot_pred_3d(1, 8, i), joint_y+plot_pred_3d(2, 8, i),...
                        joint_z+plot_pred_3d(3, 8, i));
    set(rhand, 'Faces', rhand_patch.faces, 'Vertices', rhand_patch.vertices);
       
    view([90, 13])
    
    hold off;
    drawnow;
%     toc;
    a = toc;
    pause(0.1-a);
end

end

