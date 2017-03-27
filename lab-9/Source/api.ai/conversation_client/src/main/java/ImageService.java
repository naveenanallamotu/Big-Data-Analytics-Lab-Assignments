import org.json.JSONArray;
import org.json.JSONObject;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.BufferedReader;
import java.io.IOException;

/**
 * Created by Jyothi Kiran on 13-03-2017.
 */
@WebServlet(name = "ImageService", urlPatterns = "/ImageService")
public class ImageService extends HttpServlet {

    @Override
    protected void doPost(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
        StringBuilder buffer = new StringBuilder();
        BufferedReader reader = req.getReader();
        String response="";
        String line;
        while ((line = reader.readLine()) != null) {
            buffer.append(line);
        }
        String data = buffer.toString();
        System.out.println(data);
        String output = "";
        JSONObject params = new JSONObject(data);
        JSONObject result = params.getJSONObject("result");
        JSONObject parameters = result.getJSONObject("parameters");
        if (parameters.get("pathway").toString().equals("roads")) {
            JSONObject jsonObject = new JSONObject();
            JSONArray jsonArray = new JSONArray();
            jsonArray.put("http://www.americantrails.org/photoGalleries/cool/5images/02.jpg");
            jsonArray.put("https://thumbs.dreamstime.com/z/walkway-pathway-sign-poll-to-roadside-narrow-concrete-way-along-street-surrounding-large-tropical-trees-jungle-75755404.jpg");
            jsonArray.put("http://l7.alamy.com/zooms/6ca6dc16c4584668a9169a97c4a469c2/raised-wooden-boardwalk-kirby-storter-roadside-park-florida-united-acx5t3.jpg");
            jsonArray.put("https://previews.123rf.com/images/ragsac/ragsac0706/ragsac070600034/1171802-River-crossing-with-a-stepping-stone-Stock-Photo-stones.jpg");
            jsonArray.put("http://wallpapers.ae/wp-content/uploads/2014/12/Beautiful-Pathway-Wallpaper.jpg");
            jsonArray.put("http://cdn.pcwallart.com/images/beautiful-pathway-wallpaper-3.jpg");
            jsonObject.put("data", jsonArray);
            output = jsonObject.toString();
            Data data_ob = Data.getInstance();
            data_ob.setData(output);
            data_ob.setFlag(true);
            JSONObject js = new JSONObject();
            js.put("speech", "pathways are displayed");
            js.put("displayText", "pathways are displayed");
            js.put("source", "image database");
            response = js.toString();
        }
        else if (parameters.get("vehicles").toString().equals("vehicles")) {
            JSONObject jsonObject = new JSONObject();
            JSONArray jsonArray = new JSONArray();
            jsonArray.put("http://fuelcelltoday.com/media/1844541/clarity_489x250.jpg");
            jsonArray.put("http://www.venetolowcost.com/wp-content/uploads/2015/09/KU_221_%5E_RTA_Emergency_Response_Iveco_Turbo_Daily_-_Flickr_-_Highway_Patrol_Images.jpg");
            jsonArray.put("http://www.wallpapersdb.org/wallpapers/cars/car_on_the_road_1600x900.jpg");
            jsonArray.put("http://farm8.staticflickr.com/7329/9534683012_817bb1d968_o.jpg");
            jsonArray.put("https://www.transportation.gov/sites/dot.dev/files/pictures/TrucksOnTheRoad.jpg");
            jsonArray.put("https://s-media-cache-ak0.pinimg.com/736x/b8/8a/e5/b88ae502467ecd8b5041d42f0c68fa09.jpg");
            jsonObject.put("data", jsonArray);
            output = jsonObject.toString();
            Data data_ob = Data.getInstance();
            data_ob.setData(output);
            data_ob.setFlag(true);
            JSONObject js = new JSONObject();
            js.put("speech", "Vehicles are displayed");
            js.put("displayText", "Vehicles are displayed");
            js.put("source", "image database");
            response = js.toString();
        }
        else if (parameters.get("crossing").toString().equals("crossing")){
            JSONObject jsonObject = new JSONObject();
            JSONArray jsonArray = new JSONArray();
            jsonArray.put("http://www.canadiannaturephotographer.com/Road_Photography/_DSC2348.jpg");
            jsonArray.put("http://www.stzelephants.org/step/wp-content/uploads/2012/01/ele-group-2-crossing-road.jpg");
            jsonArray.put("http://dcameragroup.com/wp-content/uploads/2014/12/deer-crossing-road.jpg");
            jsonArray.put("https://previews.123rf.com/images/smithore/smithore1204/smithore120400108/13315716-TOKYO-NOVEMBER-25-People-crossing-street-at-Hachiko-crossroad-in-Shibuya-district-on-November-25-201-Stock-Photo.jpg");
            jsonArray.put("https://cdn.shutterstock.com/shutterstock/videos/8353006/thumb/1.jpg");
            jsonArray.put("https://qph.ec.quoracdn.net/main-qimg-d6d264ff6368550886ebb254c84be292-c");
            jsonObject.put("data", jsonArray);
            output = jsonObject.toString();
            Data data_ob = Data.getInstance();
            data_ob.setData(output);
            data_ob.setFlag(true);
            JSONObject js = new JSONObject();
            js.put("speech", "Crossing Images are displayed");
            js.put("displayText", "Crossing Images are displayed");
            js.put("source", "image database");
            response = js.toString();
        }
        else if (parameters.get("buildings").toString().equals("buildings")){
            JSONObject jsonObject = new JSONObject();
            JSONArray jsonArray = new JSONArray();
            jsonArray.put("http://static1.businessinsider.com/image/530ce0b26da811733efadc58/the-worlds-most-spectacular-university-buildings.jpg");
            jsonArray.put("http://static2.businessinsider.com/image/578fc748dd08955d018b4a66/the-22-most-beautiful-buildings-in-the-world-according-to-architects.jpg");
            jsonArray.put("http://info.umkc.edu/facultyorientation/wp-content/uploads/2012/05/UMKC-Stairs-22.jpg");
            jsonArray.put("http://info.umkc.edu/unews/wp-content/uploads/2010/10/epperson-afraid_025.jpg");
            jsonArray.put("http://info.umkc.edu/unews/wp-content/uploads/2010/09/IMG_9826.jpg");
            jsonArray.put("http://iidamidamerica.org/wordpress/wp-content/uploads/2014/07/11019_00_UMKC_Bloch_Hall_N4_medium.jpg");
            jsonObject.put("data", jsonArray);
            output = jsonObject.toString();
            Data data_ob = Data.getInstance();
            data_ob.setData(output);
            data_ob.setFlag(true);
            JSONObject js = new JSONObject();
            js.put("speech", "Buildings are displayed");
            js.put("displayText", "Buildings are displayed");
            js.put("source", "image database");
            response = js.toString();
        }
        else if (parameters.get("null").toString().equals("clear")){
            Data data_ob = Data.getInstance();
            JSONObject js1 = new JSONObject();
            JSONArray jsonArray = new JSONArray();
            jsonArray.put(" ");
            js1.put("data", jsonArray);
            data_ob.setData(js1.toString());
            data_ob.setFlag(true);
            JSONObject js = new JSONObject();
            js.put("speech", "screen is cleared");
            js.put("displayText", "screen is cleared");
            js.put("source", "image database");
            response = js.toString();
        }
        resp.setHeader("Content-type", "application/json");
        resp.getWriter().write(response);
    }
}
